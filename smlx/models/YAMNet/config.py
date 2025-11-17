#!/usr/bin/env python3
# Copyright © 2025 SMLX Project

"""
YAMNet Configuration.

YAMNet is a MobileNet-v1 based audio classifier trained on AudioSet
with 521 event classes. It uses mel spectrograms as input features.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class YAMNetConfig:
    """Configuration for YAMNet audio classifier.

    YAMNet uses a MobileNet-v1 architecture with depthwise-separable
    convolutions for efficiency. It processes mel spectrograms and
    outputs probabilities for 521 AudioSet classes.
    """

    # Model architecture
    model_type: str = "yamnet"
    num_classes: int = 521  # AudioSet classes
    embedding_size: int = 1024  # Audio embedding dimension

    # Audio processing
    sample_rate: int = 16000  # Fixed for YAMNet
    patch_hop_seconds: float = 0.48  # Hop between patches
    patch_window_seconds: float = 0.96  # Window size for patches

    # Mel spectrogram parameters
    num_mel_bins: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0

    # STFT parameters
    stft_window_seconds: float = 0.025  # 25ms windows
    stft_hop_seconds: float = 0.010  # 10ms hop

    # MobileNet parameters
    depth_multiplier: float = 1.0  # For depthwise convolutions

    def __post_init__(self):
        """Validate configuration."""
        if self.sample_rate != 16000:
            raise ValueError("YAMNet only supports 16kHz sample rate")

        if self.num_classes != 521:
            raise ValueError("YAMNet has 521 AudioSet classes")

    @property
    def patch_frames(self) -> int:
        """Number of mel spectrogram frames per patch."""
        window_length_samples = int(
            round(self.stft_window_seconds * self.sample_rate)
        )
        hop_length_samples = int(
            round(self.stft_hop_seconds * self.sample_rate)
        )
        patch_window_length_samples = int(
            round(self.patch_window_seconds * self.sample_rate)
        )
        return int(1 + (patch_window_length_samples - window_length_samples) // hop_length_samples)

    @property
    def patch_bands(self) -> int:
        """Number of mel bands in patches."""
        return self.num_mel_bins

    @property
    def stft_window_length_samples(self) -> int:
        """STFT window length in samples."""
        return int(round(self.stft_window_seconds * self.sample_rate))

    @property
    def stft_hop_length_samples(self) -> int:
        """STFT hop length in samples."""
        return int(round(self.stft_hop_seconds * self.sample_rate))

    @property
    def patch_hop_length_samples(self) -> int:
        """Patch hop length in samples."""
        return int(round(self.patch_hop_seconds * self.sample_rate))

    @property
    def patch_window_length_samples(self) -> int:
        """Patch window length in samples."""
        return int(round(self.patch_window_seconds * self.sample_rate))


# Default configuration
DEFAULT_CONFIG = YAMNetConfig()


# AudioSet class names mapping
# Full list from: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
AUDIOSET_CLASSES = [
    "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell",
    "Children shouting", "Screaming", "Whispering", "Laughter", "Baby laughter",
    "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing",
    "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Singing", "Choir",
    "Yodeling", "Chant", "Mantra", "Child singing", "Synthetic singing", "Rapping",
    "Humming", "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring",
    "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff", "Run",
    "Shuffle", "Walk, footsteps", "Chewing, mastication", "Biting", "Gargling",
    "Stomach rumble", "Burping, eructation", "Hiccup", "Fart", "Hands",
    "Finger snapping", "Clapping", "Heart sounds, heartbeat", "Heart murmur",
    "Cheering", "Applause", "Chatter", "Crowd", "Hubbub, speech noise, speech babble",
    "Children playing", "Animal", "Domestic animals, pets", "Dog", "Bark", "Yip",
    "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Cat", "Purr", "Meow", "Hiss",
    "Caterwaul", "Livestock, farm animals, working animals", "Horse", "Clip-clop",
    "Neigh, whinny", "Cattle, bovinae", "Moo", "Cowbell", "Pig", "Oink", "Goat",
    "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo",
    "Turkey", "Gobble", "Duck", "Quack", "Goose", "Honk", "Wild animals",
    "Roaring cats (lions, tigers)", "Roar", "Bird", "Bird vocalization, bird call, bird song",
    "Chirp, tweet", "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot",
    "Bird flight, flapping wings", "Canidae, dogs, wolves", "Rodents, rats, mice",
    "Mouse", "Patter", "Insect", "Cricket", "Mosquito", "Fly, housefly", "Buzz",
    "Bee, wasp, etc.", "Frog", "Croak", "Snake", "Rattle", "Whale vocalization",
    "Music", "Musical instrument", "Plucked string instrument", "Guitar",
    "Electric guitar", "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar",
    "Tapping (guitar technique)", "Strum", "Banjo", "Sitar", "Mandolin", "Zither",
    "Ukulele", "Keyboard (musical)", "Piano", "Electric piano", "Organ",
    "Electronic organ", "Hammond organ", "Synthesizer", "Sampler", "Harpsichord",
    "Percussion", "Drum kit", "Drum machine", "Drum", "Snare drum", "Rimshot",
    "Drum roll", "Bass drum", "Timpani", "Tabla", "Cymbal", "Hi-hat", "Wood block",
    "Tambourine", "Rattle (instrument)", "Maraca", "Gong", "Tubular bells",
    "Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone",
    "Steelpan", "Orchestra", "Brass instrument", "French horn", "Trumpet",
    "Trombone", "Bowed string instrument", "String section", "Violin, fiddle",
    "Pizzicato", "Cello", "Double bass", "Wind instrument, woodwind instrument",
    "Flute", "Saxophone", "Clarinet", "Harp", "Bell", "Church bell", "Jingle bell",
    "Bicycle bell", "Tuning fork", "Chime", "Wind chime", "Change ringing (campanology)",
    "Harmonica", "Accordion", "Bagpipes", "Didgeridoo", "Shofar", "Theremin",
    "Singing bowl", "Scratching (performance technique)", "Pop music", "Hip hop music",
    "Beatboxing", "Rock music", "Heavy metal", "Punk rock", "Grunge",
    "Progressive rock", "Rock and roll", "Psychedelic rock", "Rhythm and blues",
    "Soul music", "Reggae", "Country", "Swing music", "Bluegrass", "Funk",
    "Folk music", "Middle Eastern music", "Jazz", "Disco", "Classical music",
    "Opera", "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass",
    "Electronica", "Electronic dance music", "Ambient music", "Trance music",
    "Music of Latin America", "Salsa music", "Flamenco", "Blues", "Music for children",
    "New-age music", "Vocal music", "A capella", "Music of Africa", "Afrobeat",
    "Christian music", "Gospel music", "Music of Asia", "Carnatic music",
    "Music of Bollywood", "Ska", "Traditional music", "Independent music", "Song",
    "Background music", "Theme music", "Jingle (music)", "Soundtrack music",
    "Lullaby", "Video game music", "Christmas music", "Dance music", "Wedding music",
    "Happy music", "Sad music", "Tender music", "Exciting music", "Angry music",
    "Scary music", "Wind", "Rustling leaves", "Wind noise (microphone)",
    "Thunderstorm", "Thunder", "Water", "Rain", "Raindrop", "Rain on surface",
    "Stream", "Waterfall", "Ocean", "Waves, surf", "Steam", "Gurgling", "Fire",
    "Crackle", "Vehicle", "Boat, Water vehicle", "Sailboat, sailing ship",
    "Rowboat, canoe, kayak", "Motorboat, speedboat", "Ship", "Motor vehicle (road)",
    "Car", "Vehicle horn, car horn, honking", "Toot", "Car alarm",
    "Power windows, electric windows", "Skidding", "Tire squeal", "Car passing by",
    "Race car, auto racing", "Truck", "Air brake", "Air horn, truck horn",
    "Reversing beeps", "Ice cream truck, ice cream van", "Bus", "Emergency vehicle",
    "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
    "Motorcycle", "Traffic noise, roadway noise", "Rail transport", "Train",
    "Train whistle", "Train horn", "Railroad car, train wagon", "Train wheels squealing",
    "Subway, metro, underground", "Aircraft", "Aircraft engine", "Jet engine",
    "Propeller, airscrew", "Helicopter", "Fixed-wing aircraft, airplane", "Bicycle",
    "Skateboard", "Engine", "Light engine (high frequency)", "Dental drill, dentist's drill",
    "Lawn mower", "Chainsaw", "Medium engine (mid frequency)", "Heavy engine (low frequency)",
    "Engine knocking", "Engine starting", "Idling", "Accelerating, revving, vroom",
    "Door", "Doorbell", "Ding-dong", "Sliding door", "Slam", "Knock", "Tap",
    "Squeak", "Cupboard open or close", "Drawer open or close",
    "Dishes, pots, and pans", "Cutlery, silverware", "Chopping (food)",
    "Frying (food)", "Microwave oven", "Blender", "Water tap, faucet",
    "Sink (filling or washing)", "Bathtub (filling or washing)", "Hair dryer",
    "Toilet flush", "Toothbrush", "Electric toothbrush", "Vacuum cleaner",
    "Zipper (clothing)", "Keys jangling", "Coin (dropping)", "Scissors",
    "Electric shaver, electric razor", "Shuffling cards", "Typing", "Typewriter",
    "Computer keyboard", "Writing", "Alarm", "Telephone", "Telephone bell ringing",
    "Ringtone", "Telephone dialing, DTMF", "Dial tone", "Busy signal", "Alarm clock",
    "Siren", "Civil defense siren", "Buzzer", "Smoke detector, smoke alarm",
    "Fire alarm", "Foghorn", "Whistle", "Steam whistle", "Mechanisms",
    "Ratchet, pawl", "Clock", "Tick", "Tick-tock", "Gears", "Pulleys",
    "Sewing machine", "Mechanical fan", "Air conditioning", "Cash register",
    "Printer", "Camera", "Single-lens reflex camera", "Tools", "Hammer",
    "Jackhammer", "Sawing", "Filing (rasp)", "Sanding", "Power tool", "Drill",
    "Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire",
    "Cap gun", "Fireworks", "Firecracker", "Burst, pop", "Eruption", "Boom",
    "Wood", "Chop", "Splinter", "Crack", "Glass", "Chink, clink", "Shatter",
    "Liquid", "Splash, splatter", "Slosh", "Squish", "Drip", "Pour",
    "Trickle, dribble", "Gush", "Fill (with liquid)", "Spray", "Pump (liquid)",
    "Stir", "Boiling", "Sonar", "Arrow", "Whoosh, swoosh, swish", "Thump, thud",
    "Thunk", "Electronic tuner", "Effects unit", "Chorus effect", "Basketball bounce",
    "Bang", "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking",
    "Bouncing", "Whip", "Flap", "Scratch", "Scrape", "Rub", "Roll", "Crushing",
    "Crumpling, crinkling", "Tearing", "Beep, bleep", "Ping", "Ding", "Clang",
    "Squeal", "Creak", "Rustle", "Whir", "Clatter", "Sizzle", "Clicking",
    "Clickety-clack", "Rumble", "Plop", "Jingle, tinkle", "Hum", "Zing", "Boing",
    "Crunch", "Silence", "Sine wave", "Harmonic", "Chirp tone", "Sound effect",
    "Pulse", "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural", "Reverberation",
    "Echo", "Noise", "Environmental noise", "Static", "Mains hum", "Distortion",
    "Sidetone", "Cacophony", "White noise", "Pink noise", "Throbbing",
    "Vibration", "Television", "Radio", "Field recording",
]


__all__ = [
    "YAMNetConfig",
    "DEFAULT_CONFIG",
    "AUDIOSET_CLASSES",
]
