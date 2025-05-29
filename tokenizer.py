import pretty_midi
import numpy as np

# -------------------------
# Vocabulary settings
# -------------------------

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = 128
VELOCITY_OFFSET = 256
TIME_SHIFT_OFFSET = 288

NUM_PITCHES = 128
NUM_VELOCITY_BINS = 32
NUM_TIME_SHIFTS = 100  # 10ms to 1000ms (in 10ms steps)

TIME_SHIFT_RESOLUTION = 10  # ms per TIME_SHIFT step
MAX_SHIFT_MS = NUM_TIME_SHIFTS * TIME_SHIFT_RESOLUTION

TOTAL_VOCAB_SIZE = TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS

# -------------------------
# Helper functions
# -------------------------

def velocity_to_bin(vel):
    return int((vel / 128) * NUM_VELOCITY_BINS)

def bin_to_velocity(bin_id):
    return int((bin_id / NUM_VELOCITY_BINS) * 128)

# -------------------------
# Encoding: MIDI → tokens
# -------------------------

def midi_to_tokens(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    events = []

    # Gather all notes from all instruments
    all_notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, 'on', note.pitch, note.velocity))
            all_notes.append((note.end, 'off', note.pitch, note.velocity))

    # Sort by time
    all_notes.sort()

    last_time = 0
    current_velocity = 64  # default

    for time, typ, pitch, velocity in all_notes:
        # Add time shift if needed
        delta_time = time - last_time
        ms = int(delta_time * 1000)

        while ms > 0:
            shift_amount = min(ms, MAX_SHIFT_MS)
            shift_token = TIME_SHIFT_OFFSET + (shift_amount // TIME_SHIFT_RESOLUTION)
            events.append(shift_token)
            ms -= shift_amount

        if typ == 'on':
            vel_bin = velocity_to_bin(velocity)
            events.append(VELOCITY_OFFSET + vel_bin)
            events.append(NOTE_ON_OFFSET + pitch)
        elif typ == 'off':
            events.append(NOTE_OFF_OFFSET + pitch)

        last_time = time

    return events

# -------------------------
# Decoding: tokens → MIDI
# -------------------------

def tokens_to_midi(tokens, output_path='output.mid'):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    midi.instruments.append(piano)

    time = 0.0
    current_velocity = 64
    note_dict = {}

    for token in tokens:
        if NOTE_ON_OFFSET <= token < NOTE_OFF_OFFSET:
            pitch = token - NOTE_ON_OFFSET
            note_dict[pitch] = time
        elif NOTE_OFF_OFFSET <= token < VELOCITY_OFFSET:
            pitch = token - NOTE_OFF_OFFSET
            if pitch in note_dict:
                start = note_dict[pitch]
                end = time
                note = pretty_midi.Note(velocity=current_velocity, pitch=pitch, start=start, end=end)
                piano.notes.append(note)
                del note_dict[pitch]
        elif VELOCITY_OFFSET <= token < TIME_SHIFT_OFFSET:
            vel_bin = token - VELOCITY_OFFSET
            current_velocity = bin_to_velocity(vel_bin)
        elif TIME_SHIFT_OFFSET <= token < TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS:
            shift = (token - TIME_SHIFT_OFFSET + 1) * TIME_SHIFT_RESOLUTION / 1000.0
            time += shift

    midi.write(output_path)
    print(f"Saved MIDI to {output_path}")

# -------------------------
# Test
# -------------------------

if __name__ == "__main__":
    # Example usage
    tokens = midi_to_tokens("example.mid")  # Replace with your file
    print("Tokens:", tokens[:50])

    tokens_to_midi(tokens, "reconstructed.mid")
