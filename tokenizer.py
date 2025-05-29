import pretty_midi
import numpy as np

# -------------------------
# Vocabulary settings
# -------------------------

NOTE_ON_OFFSET = 0
NOTE_OFF_OFFSET = 128
VELOCITY_OFFSET = 256
TIME_SHIFT_OFFSET = 384  # changed due to larger velocity vocab

NUM_PITCHES = 128
NUM_VELOCITY_BINS = 128  # no quantization loss now
NUM_TIME_SHIFTS = 1000  # 1ms to 1000ms

TIME_SHIFT_RESOLUTION = 1  # now using 1ms
MAX_SHIFT_MS = NUM_TIME_SHIFTS * TIME_SHIFT_RESOLUTION

TOTAL_VOCAB_SIZE = TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS

def velocity_to_bin(vel):
    return int(np.clip(vel, 1, 127))

def bin_to_velocity(bin_id):
    return int(np.clip(bin_id, 1, 127))

# -------------------------
# Encoding: MIDI → tokens
# -------------------------

def midi_to_tokens(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Optional: strip sustain pedal CCs to simplify
    for instrument in midi.instruments:
        instrument.control_changes = []

    events = []
    all_notes = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, 'on', note.pitch, note.velocity))
            all_notes.append((note.end, 'off', note.pitch, note.velocity))

    all_notes.sort()
    last_time = 0.0
    current_velocity = 64

    for time, typ, pitch, velocity in all_notes:
        delta_time = time - last_time
        ms = int(round(delta_time * 1000))

        while ms > 0:
            shift_amount = min(ms, MAX_SHIFT_MS)
            shift_token = TIME_SHIFT_OFFSET + (shift_amount // TIME_SHIFT_RESOLUTION)
            events.append(shift_token)
            ms -= shift_amount

        if typ == 'on':
            vel_bin = velocity_to_bin(velocity)
            events.append(VELOCITY_OFFSET + vel_bin)
            events.append(NOTE_ON_OFFSET + pitch)
            print(f"[ENCODE] NOTE_ON {pitch} vel={velocity}")
        elif typ == 'off':
            events.append(NOTE_OFF_OFFSET + pitch)
            print(f"[ENCODE] NOTE_OFF {pitch}")

        last_time = time

    return events

# -------------------------
# Decoding: tokens → MIDI
# -------------------------

def tokens_to_midi(tokens, output_path='output_refined.mid'):
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
                note = pretty_midi.Note(velocity=min(current_velocity + 15, 127), pitch=pitch, start=start, end=end)
                piano.notes.append(note)
                del note_dict[pitch]
                print(f"[DECODE] NOTE {pitch} from {start:.3f}s to {end:.3f}s, vel={current_velocity}")
        elif VELOCITY_OFFSET <= token < TIME_SHIFT_OFFSET:
            vel_bin = token - VELOCITY_OFFSET
            current_velocity = bin_to_velocity(vel_bin)
        elif TIME_SHIFT_OFFSET <= token < TOTAL_VOCAB_SIZE:
            shift = (token - TIME_SHIFT_OFFSET + 1) * TIME_SHIFT_RESOLUTION / 1000.0
            time += shift

    midi.write(output_path)
    print(f"Saved refined MIDI to {output_path}")

# -------------------------
# Quick Test
# -------------------------

if __name__ == "__main__":
    tokens = midi_to_tokens("example (1).mid")
    print("Total tokens:", len(tokens))
    tokens_to_midi(tokens, "reconstructed_refined.mid")
