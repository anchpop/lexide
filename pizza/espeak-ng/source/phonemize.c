// SPDX-License-Identifier: GPL-3.0-or-later
#include <emscripten.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <espeak-ng/encoding.h>
#include <espeak-ng/espeak_ng.h>
#include "dictionary.h"
#include "intonation.h"
#include "setlengths.h"
#include "synthdata.h"
#include "voice.h"

// A language switch can occur inside a single word. Let espeak's dictionary
// lookup request its exact dependency while its translation state is suspended.
EM_ASYNC_JS(int, fetch_dictionary, (const char *name), {
    return await Module['loadDictionary'](UTF8ToString(name));
});

int __real_LoadDictionary(Translator *tr, const char *name, int quiet);
int __wrap_LoadDictionary(Translator *tr, const char *name, int quiet)
{
    char path[128];
    snprintf(path, sizeof(path), "/espeak-ng-data/%s_dict", name);
    if (access(path, F_OK) && !fetch_dictionary(name)) return 1;
    return __real_LoadDictionary(tr, name, quiet);
}

int initialize(const char *name)
{
    espeak_ng_InitializePath("/espeak-ng-data");
    int status = espeak_ng_Initialize(NULL);
    if (status != ENS_OK) return status;
    p_decoder = create_text_decoder();
    if (!p_decoder) return -1;
    status = espeak_ng_SetVoiceByName(name);
    if (status == ENS_OK) return status;
    espeak_VOICE selector = {.languages = name};
    return espeak_ng_SetVoiceByProperties(&selector);
}

char *phonemize(const char *text)
{
    // Match the CLI's sync_espeak_Synth and SpeakNextClause through the
    // phoneme trace. The waveform generator is unnecessary for this output.
    InitText(espeakPHONEMES | espeakENDPAUSE);
    option_phoneme_input = espeakPHONEMES;
    option_endpause = espeakENDPAUSE;
    if (text_decoder_decode_string_multibyte(p_decoder, text, translator->encoding, espeakCHARS_AUTO) != ENS_OK)
        return NULL;

    size_t length;
    char *result;
    FILE *out = open_memstream(&result, &length);
    if (!out) return NULL;
    while (!text_decoder_eof(p_decoder)) {
        int tone;
        char *voice_change;
        SelectPhonemeTable(voice->phoneme_tab_ix);
        TranslateClause(translator, &tone, &voice_change);
        CalcPitches(translator, tone);
        CalcLengths(translator);
        fprintf(out, "%s\n", GetTranslatedPhonemeString(espeakPHONEMES_IPA | ('_' << 8)));
    }
    if (fclose(out)) {
        free(result);
        return NULL;
    }
    return result;
}
