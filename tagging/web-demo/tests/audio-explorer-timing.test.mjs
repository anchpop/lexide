import test from "node:test";
import assert from "node:assert/strict";
import { AudioExplorer } from "../www/audio-explorer.js";

// Minimal DOM doubles exercise the real track labels, positions and selection.
function element() {
  return {style: {}, children: [], value: "1", clientWidth: 400, hidden: false,
    classList: {toggle() {}}, setAttribute() {}, replaceChildren() { this.children = []; },
    append(child) { this.children.push(child); },
    getContext: () => ({fillRect() {}, drawImage() {}})};
}

test("phoneme track labels, widths and seeking share the declared frame interval", () => {
  const nodes = new Map();
  const explorer = Object.create(AudioExplorer.prototype);
  explorer.$ = id => { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); };
  explorer.duration = 2;
  explorer.onPhone = () => {};
  explorer.update = () => {};
  explorer.selected = -1;
  const oldDocument = globalThis.document;
  globalThis.document = {createElement: element};
  try {
    explorer.show([{phoneme: "a", startFrame: 2, endFrame: 5}], .04);
    const button = explorer.$("phoneme-track").children[0];
    assert.match(button.title, /0.08–0.20s/);
    assert.equal(button.style.left, "16px");
    assert.equal(button.style.width, "24px");
    let followed;
    explorer.ensureVisible = value => { followed = value; };
    explorer.select(0, true);
    assert.equal(followed, .08);
    assert.equal(explorer.$("selected-region").style.left, "16px");
    assert.equal(explorer.$("selected-region").style.width, "24px");
    assert.throws(() => explorer.show([], undefined), /Invalid frame interval/);
  } finally { globalThis.document = oldDocument; }
});

test("keyboard audio navigation works before inference without fabricating phone timing", () => {
  const previous = {document: globalThis.document, window: globalThis.window, ResizeObserver: globalThis.ResizeObserver};
  const nodes = new Map();
  globalThis.document = {getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  }};
  globalThis.window = {addEventListener() {}};
  globalThis.ResizeObserver = class { observe() {} };
  try {
    const audio = {currentTime: .5, addEventListener() {}};
    const explorer = new AudioExplorer({audio, onPhone() {}, onFrame() {}});
    explorer.duration = 2;
    let sought;
    explorer.seek = value => { sought = value; };
    const key = key => nodes.get("spectrogram").onkeydown({key, preventDefault() {}});
    key("ArrowRight"); assert.equal(sought, .52);
    key("Home"); assert.equal(sought, 0);
    key("End"); assert.equal(sought, 2);
    assert.equal(explorer.frameSeconds, null);
    explorer.frameSeconds = .04;
    key("ArrowRight"); assert.equal(sought, .54);
  } finally { Object.assign(globalThis, previous); }
});
