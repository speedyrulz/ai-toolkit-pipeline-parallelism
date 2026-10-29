import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getModelsPath } from '@/server/settings';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MODEL_EXTS = ['.safetensors', '.sft', '.gguf', '.ckpt', '.pt', '.pth', '.bin'];
// a folder holding one of these is a loadable model (diffusers pipeline or transformers checkpoint)
const MODEL_DIR_MARKERS = ['model_index.json', 'config.json'];

const isDir = async (p: string) => {
  try {
    return (await fs.promises.stat(p)).isDirectory();
  } catch {
    return false;
  }
};

const listRoots = async (): Promise<string[]> => {
  if (process.platform !== 'win32') return ['/'];
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  const found = await Promise.all(letters.map(async l => ((await isDir(`${l}:\\`)) ? `${l}:\\` : null)));
  return found.filter((r): r is string => !!r);
};

// Server-side folder browser for picking a local model. The UI may run on a
// different machine than the trainer, so a native file dialog can't return a
// path the backend could load.
export async function GET(request: NextRequest) {
  const modelsPath = path.resolve(await getModelsPath());
  const requested = request.nextUrl.searchParams.get('path')?.trim();

  let dir = requested ? path.resolve(requested) : modelsPath;
  // a file path (e.g. the current field value) opens its containing folder
  if (!(await isDir(dir))) {
    const parent = path.dirname(dir);
    dir = (await isDir(parent)) ? parent : (await isDir(modelsPath)) ? modelsPath : path.resolve(process.cwd(), '..');
  }

  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(dir, { withFileTypes: true });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'Unable to read folder' }, { status: 400 });
  }

  const dirs: { name: string; path: string; isModel: boolean }[] = [];
  const files: { name: string; path: string; size: number }[] = [];
  await Promise.all(
    entries.map(async e => {
      if (e.name.startsWith('.') || e.name.startsWith('$')) return;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) {
        const isModel = MODEL_DIR_MARKERS.some(m => fs.existsSync(path.join(p, m)));
        dirs.push({ name: e.name, path: p, isModel });
      } else if (e.isFile() && MODEL_EXTS.includes(path.extname(e.name).toLowerCase())) {
        try {
          const st = await fs.promises.stat(p);
          files.push({ name: e.name, path: p, size: st.size });
        } catch {}
      }
    }),
  );
  dirs.sort((a, b) => a.name.localeCompare(b.name));
  files.sort((a, b) => a.name.localeCompare(b.name));

  const parent = path.dirname(dir);
  return NextResponse.json(
    {
      path: dir,
      parent: parent !== dir ? parent : null,
      isModel: MODEL_DIR_MARKERS.some(m => fs.existsSync(path.join(dir, m))),
      dirs,
      files,
      roots: await listRoots(),
      modelsPath,
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
