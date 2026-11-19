// The pokerlab page reads the HTML5 game (js-css-poker) through the browser's own window capture: "창 추가" calls
// getDisplayMedia and Chromium picks the game tab by its title, as a person would in the picker (record.py feeds
// the page a canvas instead). The game is played meanwhile; the reader is "정확하게" first, then "빠르게".
// Writes OUT_DIR/log.json (every request the page made, with its time and answer), truth.json (the game's own
// state each second), lines.json (the page's status line) and page.png.
//
//   python -m pokerlab serve --port 8790 --record OUT_DIR/hands.jsonl      (with a reading key)
//   python -I -m http.server 8781 --directory GAME_DIR                      (js-css-poker: poker.html)
//   xvfb-run -a node capture_check.js OUT_DIR ../fixtures/html5_notes.txt 2 1.5       (Linux without a screen)
//   node capture_check.js OUT_DIR ../fixtures/html5_notes.txt 2 1.5                   (Windows, macOS)
//
// Real reads: about $0.2 a minute as "정확하게", about $1 a minute as "빠르게" (it reads more often).
// The tab capture needs a browser with a window: Chromium in headless mode does not offer it.
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [out, notesFile, bestMin, fastMin] = process.argv.slice(2);
  const notes = fs.readFileSync(notesFile, 'utf8').trim();
  const browser = await chromium.launch({ headless: false, args: [
    '--auto-select-tab-capture-source-by-title=The Poker Game', '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const game = await ctx.newPage();
  await game.goto('http://127.0.0.1:8781/poker.html');
  await game.waitForTimeout(1500);
  await game.mouse.click(628, 86);                       // five opponents
  await game.waitForTimeout(1000);
  await game.selectOption('#speed-options', '0');
  const t0 = Date.now(), at = () => (Date.now() - t0) / 1000;
  const truth = [];
  const truthTimer = setInterval(async () => {
    try {
      truth.push({ t: at(), state: await game.evaluate(() => ({
        players: players.map(p => ({ name: p.name, bankroll: p.bankroll, cards: [p.carda, p.cardb], status: p.status, bet: p.subtotal_bet, total: p.total_bet })),
        board: board.filter(Boolean), button: button_index, pot: get_pot_size() })) });
    } catch (e) {}
  }, 1000);

  const ui = await ctx.newPage();
  await ui.addInitScript(() => {                         // every request the page makes, timed
    window.__log = [];
    const send = window.fetch.bind(window);
    window.fetch = async (url, options) => {
      const body = options && options.body ? JSON.parse(options.body) : {};
      const entry = { url: String(url), do: body.do || 'frame', reader: body.reader, sent: performance.now() };
      const reply = await send(url, options);
      const copy = reply.clone();
      try { const data = await copy.json(); entry.checking = data.checking; entry.board = data.table && data.table.board;
        entry.spending = data.spending; entry.error = data.error; } catch (e) {}
      entry.back = performance.now(); entry.ms = Math.round(entry.back - entry.sent);
      window.__log.push(entry);
      return reply;
    };
  });
  await ui.goto('http://127.0.0.1:8790/');
  await ui.click('#modeCapture');
  await ui.click('#capAdd');                             // the browser's window picker: Chromium picks the game tab
  await ui.waitForSelector('#capSources .source', { timeout: 15000 });
  const track = await ui.evaluate(() => { const t = cap.sources[0].stream.getVideoTracks()[0]; return { label: t.label, settings: t.getSettings() }; });
  console.log('captured', JSON.stringify(track));
  await ui.locator('#capSources .source input[type=text]').first().fill('JS Poker');
  await ui.selectOption('#capSources .source select >> nth=0', '2');
  await ui.locator('#capSources .source input[type=text]').nth(1).fill(notes);
  await ui.selectOption('#capSources .source select >> nth=1', 'best');

  const lines = [];
  const lineTimer = setInterval(async () => {
    try { lines.push({ t: at(), line: await ui.locator('#capSources .source .state').first().innerText() }); } catch (e) {}
  }, 2000);
  const turn = "getComputedStyle(document.getElementById('call-button')).visibility==='visible'?document.getElementById('call-button').innerText:''";
  let decisions = 0, switched = false;
  const end = Date.now() + (Number(bestMin) + Number(fastMin)) * 60000;
  while (Date.now() < end) {
    if (!switched && Date.now() > t0 + Number(bestMin) * 60000) {
      await ui.selectOption('#capSources .source select >> nth=1', 'fast');
      switched = true; console.log('reader: fast at', at().toFixed(1));
    }
    const label = await game.evaluate(turn).catch(() => '');
    if (label) {
      decisions += 1;
      await game.waitForTimeout(800);
      const fold = decisions % 4 === 0 && label.toLowerCase().includes('call');
      await game.click(fold ? '#fold-button' : '#call-button', { timeout: 3000 }).catch(() => {});
    }
    await game.waitForTimeout(1500);
  }
  clearInterval(truthTimer); clearInterval(lineTimer);
  await ui.waitForTimeout(6000);                         // the last answers and readings again
  const log = await ui.evaluate(() => window.__log);
  fs.writeFileSync(`${out}/log.json`, JSON.stringify(log, null, 1));
  fs.writeFileSync(`${out}/truth.json`, JSON.stringify(truth));
  fs.writeFileSync(`${out}/lines.json`, JSON.stringify(lines, null, 1));
  await ui.screenshot({ path: `${out}/page.png`, fullPage: false });
  console.log('requests', log.length, 'last line', lines.length ? lines[lines.length - 1].line : '');
  await browser.close();
})();
