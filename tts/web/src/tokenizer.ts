export interface Gram {
  words: readonly string[]
  pos: readonly string[]
  count: number
  phonemes: string
}

export interface Frontend {
  version: number
  symbolTable: string[]
  phones: string[]
  punctuation: string[]
  aliases: Record<string, string>
  clitics: string[]
  elisionEndings: string[]
  attachedTags: string[]
  pos: Record<string, number>
  otherPos: number
  stress: Record<string, number>
  rarityBuckets: number
  wordBreak: string
  phrase: string
  nucleus: string
}

type Features = Partial<Record<'stress' | 'tone' | 'accent' | 'pos' | 'rarity' | 'boundary', number>>

export function tokenizer(spec: Frontend, columns: (keyof Features)[]) {
  if (spec.version !== 4) throw new Error('Unsupported tokenizer version ' + spec.version)
  const ids = new Map(spec.symbolTable.map((symbol, id) => [symbol, id]))
  const phones = new Set(spec.phones)
  const marks = new Set(spec.punctuation)
  const aliases = new Map(Object.entries(spec.aliases))
  const pos = new Map(Object.entries(spec.pos))
  function punctuation(words: readonly string[]) {
    return words.flatMap(word => {
      const alias = aliases.get(word) ?? word
      return marks.has(alias) ? [alias] : Array.from(alias, c => aliases.get(c) ?? c).filter(c => marks.has(c))
    })
  }
  function spokenWords(gram: Gram, attach: boolean) {
    const leading: string[] = []
    const spoken: {pos: number, after: string[]}[] = []
    let join: string | undefined
    if (gram.words.length !== gram.pos.length) throw new Error('Expected one POS tag per word')
    for (const [i, word] of gram.words.entries()) {
      const previous = spoken.at(-1)
      const tag = gram.pos[i]
      const tagId = pos.get(tag) ?? spec.otherPos
      if (word === '-' && previous) join = '-'
      else if (!/[\p{L}\p{N}]/u.test(word)) (previous?.after ?? leading).push(...punctuation([word]))
      else if (previous && !previous.after.length &&
          (join !== undefined || spec.clitics.some(c => word.toLowerCase().startsWith(c)) ||
           (attach && spec.attachedTags.includes(tag)))) {
        if (join === '') previous.pos = tagId
        join = undefined
      } else {
        spoken.push({pos: tagId, after: []})
        join = spec.elisionEndings.some(c => word.endsWith(c)) ? '' : undefined
      }
    }
    return {leading, spoken}
  }
  return (grams: readonly Gram[]) => {
    const tokens: number[] = [], lexical: number[][] = []
    let last = '', spokenBefore = false, firstPhone = -1
    function add(symbol: string, features: Features = {}) {
      const id = ids.get(symbol)
      if (id === undefined) throw new Error('Unknown symbol ' + symbol)
      tokens.push(id)
      lexical.push(columns.map(name => features[name] ?? 0))
      last = symbol
    }
    function boundary(punctuation: string[], kind: number) {
      for (const mark of punctuation.length ? punctuation : ['<sp>']) add(mark, {boundary: kind})
    }
    add('<bos>')
    for (const gram of grams) {
      const sequence = (gram.phonemes.match(/[^\p{White_Space}\u001c-\u001f]+/gu) ?? []).join(' ')
      if (!sequence) {
        for (const mark of punctuation(gram.words)) add(mark, {boundary: 2})
        continue
      }
      const groups = sequence.split(' ' + spec.wordBreak + ' ')
      const plain = spokenWords(gram, false)
      const {leading, spoken} = groups.length !== plain.spoken.length && spec.attachedTags.length
        ? spokenWords(gram, true) : plain
      if (leading.length || (spokenBefore && !marks.has(last))) boundary(leading, 2)
      const rarity = 1 + Math.floor(Math.log2(Math.min(2 ** (spec.rarityBuckets - 1), gram.count + 1)))
      const matched = groups.length === spoken.length
      for (const [i, group] of groups.entries()) {
        if (i) boundary(matched ? spoken[i - 1].after : [], 1)
        const tag = matched || groups.length === 1 ? spoken[i]?.pos ?? 0 : 0
        let start = 0
        for (let token of group.split(' ')) {
          if (token === spec.phrase) { start = 1; continue }
          const stress = spec.stress[token[0]] ?? 0
          if (stress) token = token.slice(1)
          const nucleus = token.endsWith(spec.nucleus)
          if (nucleus) token = token.slice(0, -spec.nucleus.length)
          const phone = token.replace(/[0-9]+$/, '')
          if (!phones.has(phone)) throw new Error('Phone outside the voice inventory: ' + phone)
          if (firstPhone === -1) firstPhone = tokens.length
          add(phone, {stress, tone: Number(token.slice(phone.length)), accent: start | (nucleus ? 2 : 0), pos: tag, rarity})
          start = 0
        }
      }
      for (const mark of spoken.at(-1)?.after ?? []) add(mark, {boundary: 2})
      spokenBefore = true
    }
    add('<eos>')
    return {ids: tokens, lexical, firstPhone}
  }
}

