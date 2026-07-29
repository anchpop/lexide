/* tslint:disable */
/* eslint-disable */

export class Parsley {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Whether the Japanese dictionary has been loaded.
     */
    has_japanese_dictionary(): boolean;
    /**
     * Install the Japanese boundary dictionary (`onnx/jpn-unidic.bin`, ~87MB) fetched by
     * the page. Optional and only affects Japanese; everything else is already exact.
     */
    load_japanese_dictionary(bytes: Uint8Array): void;
    /**
     * Build from the two safetensors artifacts (fetched by the page).
     */
    constructor(tokenizer_weights: Uint8Array, segmenter_weights: Uint8Array);
    /**
     * Sentence `[start, end)` char spans as a JSON array of pairs.
     */
    sentence_spans(text: string, lang?: string | null): string;
    /**
     * Token `[start, end)` char spans as a JSON array of pairs. `lang` is an optional
     * three-letter code (deu/eng/fra/hin/ita/jpn/kor/por/rus/spa); null = language-free.
     */
    token_spans(text: string, lang?: string | null): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_parsley_free: (a: number, b: number) => void;
    readonly parsley_has_japanese_dictionary: (a: number) => number;
    readonly parsley_load_japanese_dictionary: (a: number, b: number, c: number) => [number, number];
    readonly parsley_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly parsley_sentence_spans: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly parsley_token_spans: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
