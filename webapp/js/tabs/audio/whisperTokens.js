/**
 * I token di testo d'esempio che Whisper scrive nel tab Audio (soli dati: niente three.js né DOM, così
 * li importano la scena di Whisper, le azioni e la rete neurale del tab senza tirarsi dietro la scena).
 *
 * È l'inizio di una trascrizione come la fa Whisper: i token speciali che chiedono la lingua e il compito
 * (<|startoftranscript|>, <|it|>, <|transcribe|>, <|notimestamps|>), il testo (gli spazi iniziali fanno
 * parte del token, come nel tokenizer BPE di Whisper) e la fine (<|endoftext|>). Un esempio scritto a mano,
 * non l'uscita di un modello: la scena e la rete lo dichiarano.
 */
export const WHISPER_TOKENS = Object.freeze(["<|startoftranscript|>", "<|it|>", "<|transcribe|>", "<|notimestamps|>", " Ciao", " a", " tutti", "<|endoftext|>"]);
