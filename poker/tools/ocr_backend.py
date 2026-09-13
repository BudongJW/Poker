"""OCR backend, with a subprocess fallback when tesserocr cannot be installed.

`tesserocr` binds to Tesseract's C++ API in-process, which is fast, but it ships only as
a source distribution: there is no prebuilt wheel for any Python version, and building it
needs Tesseract's development headers and import libraries. A normal Windows Tesseract
install has the runtime and neither of those, so `pip install tesserocr` fails with
"Tesseract library not found in LIBPATH" and the whole bot is unimportable - every module
path runs through `screen_operations`.

This module keeps tesserocr when it is available and otherwise drives the `tesseract`
executable, which the same install already provides. The shim implements only the three
calls the bot makes, under tesserocr's own names so the call sites do not change.

The cost is process startup per call: about 110 ms against a few milliseconds in-process.
`batch_text()` exists because that cost is almost entirely model loading - ten images
through one invocation take 199 ms rather than 1.1 s, so callers with several images
should hand them over together.
"""

# pylint: disable=invalid-name
# Method names mirror tesserocr's PascalCase API so this is a drop-in replacement.
import logging
import os
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)

# Where Windows installers put it, since the runtime is often not on PATH.
_LIKELY_PATHS = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    '/usr/bin/tesseract',
    '/usr/local/bin/tesseract',
    '/opt/homebrew/bin/tesseract',
)


def find_tesseract():
    """Locate the tesseract executable, or None. TESSERACT_CMD overrides the search."""
    override = os.environ.get('TESSERACT_CMD')
    if override and os.path.exists(override):
        return override
    found = shutil.which('tesseract')
    if found:
        return found
    for candidate in _LIKELY_PATHS:
        if os.path.exists(candidate):
            return candidate
    return None


class TesseractCliAPI:
    """tesserocr's PyTessBaseAPI surface, backed by the tesseract executable.

    Only SetVariable / SetImage / GetUTF8Text are implemented - the three the bot calls.
    """

    def __init__(self, path=None, psm=7, oem=1, executable=None):
        self.executable = executable or find_tesseract()
        if not self.executable:
            raise RuntimeError(
                "No OCR backend: tesserocr is not importable and no tesseract "
                "executable was found. Install Tesseract, or set TESSERACT_CMD to it.")
        self.tessdata = path
        self.psm = psm
        self.oem = oem
        self.variables = {}
        self._image = None
        log.info("OCR backend: %s (subprocess; tesserocr unavailable)", self.executable)

    def SetVariable(self, name, value):
        """Queue a -c NAME=VALUE for the next run."""
        self.variables[str(name)] = str(value)
        return True

    def SetImage(self, image):
        """Hold the PIL image the next GetUTF8Text() should read."""
        self._image = image

    def GetUTF8Text(self):
        """Recognise the image set by SetImage(). Returns '' on any failure."""
        if self._image is None:
            return ''
        results = self.batch_text([self._image])
        return results[0] if results else ''

    def _argv(self, source, output='stdout'):
        argv = [self.executable, source, output, '--psm', str(self.psm),
                '--oem', str(self.oem)]
        if self.tessdata and os.path.isdir(self.tessdata):
            argv += ['--tessdata-dir', self.tessdata]
        for name, value in self.variables.items():
            argv += ['-c', f'{name}={value}']
        return argv

    def batch_text(self, images):
        """Recognise several images in one invocation, in order.

        Tesseract's per-call cost is dominated by loading the language model, so this is
        roughly 5x faster per image than calling GetUTF8Text() in a loop. Returns one
        string per image, empty where recognition failed.
        """
        images = list(images)
        if not images:
            return []
        with tempfile.TemporaryDirectory(prefix='pokerocr-') as workdir:
            paths = []
            for index, image in enumerate(images):
                path = os.path.join(workdir, f'{index:03d}.png')
                try:
                    image.save(path)
                except (OSError, ValueError, AttributeError) as exc:
                    log.warning("Could not write an image for OCR (%s)", exc)
                    return [''] * len(images)
                paths.append(path)

            if len(paths) == 1:
                source = paths[0]
            else:
                source = os.path.join(workdir, 'list.txt')
                with open(source, 'w', encoding='utf-8') as handle:
                    handle.write('\n'.join(paths))

            try:
                completed = subprocess.run(self._argv(source), capture_output=True,
                                           timeout=30, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("OCR invocation failed (%s)", exc)
                return [''] * len(images)

        if completed.returncode != 0:
            log.debug("tesseract exited %d: %s", completed.returncode,
                      completed.stderr.decode('utf-8', 'replace')[:200])
        text = completed.stdout.decode('utf-8', 'replace')
        # Multi-image runs separate pages with a form feed.
        pages = text.split('\f') if len(images) > 1 else [text]
        pages = [page for page in pages if page.strip() != ''] if len(images) > 1 else pages
        pages += [''] * (len(images) - len(pages))
        return pages[:len(images)]


def make_api(path=None, psm=None, oem=None):
    """Return an OCR API: the real tesserocr one where available, else the shim.

    Falls back rather than raising, because an unimportable OCR layer makes the entire
    bot unimportable - `screen_operations` is on every module path.
    """
    try:
        from tesserocr import PyTessBaseAPI, PSM, OEM  # pylint: disable=import-outside-toplevel
        log.info("OCR backend: tesserocr (in-process)")
        return PyTessBaseAPI(path=path,
                             psm=PSM.SINGLE_LINE if psm is None else psm,
                             oem=OEM.LSTM_ONLY if oem is None else oem)
    except ImportError:
        # tesserocr's SINGLE_LINE is 7 and LSTM_ONLY is 1; the CLI takes the same numbers.
        return TesseractCliAPI(path=path, psm=7 if psm is None else psm,
                               oem=1 if oem is None else oem)
