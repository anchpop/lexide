import assert from 'node:assert/strict'
import {createInterface} from 'node:readline'
import {tokenizer} from '../dist/tokenizer.js'

let tokenize
for await (const line of createInterface({input: process.stdin})) {
  const row = JSON.parse(line)
  if (!tokenize) { tokenize = tokenizer(...row); continue }
  const [name, grams, ids, lexical] = row
  const actual = tokenize(grams)
  assert.deepEqual(actual.ids, ids, name)
  assert.deepEqual(actual.lexical, lexical, name)
}
