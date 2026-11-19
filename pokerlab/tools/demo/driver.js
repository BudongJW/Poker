// A game window and the pokerlab page at phone width, driven over a local port by record.py.
// The game's pictures are fed to the page as the window it reads, and saved with their times, as are
// pictures of the page (its status line and its results); compose.py makes a video of them.
// node driver.js PORT OUT_DIR POKERLAB_URL
const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
(async () => {
  const port = Number(process.argv[2]), out = process.argv[3], uiUrl = process.argv[4];
  const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'] });
  const t0 = Date.now();
  const game = await (await browser.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
  const ui = await (await browser.newContext({ viewport: { width: 720, height: 1600 }, deviceScaleFactor: 1.5 })).newPage();
  await ui.addInitScript(() => {
    navigator.mediaDevices.getDisplayMedia = async () => {
      const c = document.createElement('canvas'); c.width = 1280; c.height = 800; const x = c.getContext('2d');
      x.fillStyle = '#000'; x.fillRect(0, 0, 1280, 800);
      window.__show = url => new Promise(done => { const img = new Image(); img.onload = () => { x.drawImage(img, 0, 0); done(); }; img.src = url; });
      return c.captureStream(10);
    };
  });
  for (const d of ['frames', 'state', 'result']) fs.mkdirSync(`${out}/${d}`, { recursive: true });
  const at = () => ((Date.now() - t0) / 1000).toFixed(3);
  let pump = null, uiPump = null, n = 0, m = 0;
  http.createServer(async (req, res) => {
    const u = new URL(req.url, 'http://x'), q = Object.fromEntries(u.searchParams);
    try {
      let result = 'ok';
      if (u.pathname === '/goto') await game.goto(q.url);
      else if (u.pathname === '/click') await game.mouse.click(Number(q.x), Number(q.y));
      else if (u.pathname === '/type') await game.keyboard.type(q.text);
      else if (u.pathname === '/shot') await game.screenshot({ path: q.path, timeout: 10000 });
      else if (u.pathname === '/clicksel') await game.click(q.sel, { timeout: 3000 });
      else if (u.pathname === '/select') await game.selectOption(q.sel, q.value);
      else if (u.pathname === '/eval') result = JSON.stringify(await game.evaluate(q.expr));
      else if (u.pathname === '/wait') await game.waitForTimeout(Number(q.ms));
      else if (u.pathname === '/ui_open') {
        await ui.goto(uiUrl);
        await ui.click('#capAdd');
        await ui.locator('#capSources .source input[type=text]').first().fill(q.label || '');
        await ui.selectOption('#capSources .source select >> nth=0', '2');
        await ui.locator('#capSources .source input[type=text]').nth(1).fill(q.notes || '');
      } else if (u.pathname === '/start') {
        let busy = false, uiBusy = false;
        pump = setInterval(async () => {
          if (busy) return; busy = true;
          try {
            const jpg = await game.screenshot({ type: 'jpeg', quality: 88, timeout: 5000 });
            fs.writeFileSync(`${out}/frames/${String(n++).padStart(5, '0')}_${at()}.jpg`, jpg);
            await ui.evaluate(url => window.__show && window.__show(url), 'data:image/jpeg;base64,' + jpg.toString('base64'));
          } catch (e) {}
          busy = false;
        }, 700);
        const truth = [];
        setInterval(async () => {          // the HTML5 game's own state, to check the records against afterwards
          try {
            truth.push({ t: Number(at()), state: await game.evaluate(() => ({
              players: players.map(p => ({ name: p.name, bankroll: p.bankroll, cards: [p.carda, p.cardb], status: p.status, bet: p.subtotal_bet, total: p.total_bet })),
              board: board.filter(Boolean), button: button_index, pot: get_pot_size() })) });
            fs.writeFileSync(`${out}/truth.json`, JSON.stringify(truth));
          } catch (e) {}
        }, 1000);
        uiPump = setInterval(async () => {
          if (uiBusy) return; uiBusy = true;
          try {
            const stamp = `${String(m++).padStart(5, '0')}_${at()}`;
            await ui.locator('#capSources .source .state').first().screenshot({ path: `${out}/state/${stamp}.png`, timeout: 5000 });
            const box = await ui.locator('#result').boundingBox();
            if (box) await ui.screenshot({ path: `${out}/result/${stamp}.png`, clip: { x: box.x, y: box.y, width: box.width, height: Math.min(box.height, 760) }, timeout: 5000 });
          } catch (e) {}
          uiBusy = false;
        }, 1000);
      } else if (u.pathname === '/state') result = await ui.locator('#capSources .source .state').first().innerText();
      else if (u.pathname === '/quit') { clearInterval(pump); clearInterval(uiPump); res.end('bye'); await browser.close(); process.exit(0); }
      res.end(String(result));
    } catch (e) { res.statusCode = 500; res.end(String(e)); }
  }).listen(port, '127.0.0.1');
  console.log('driver on', port);
})();
