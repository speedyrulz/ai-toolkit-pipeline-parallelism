import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

// Identity eval results for a job (toolkit/identity_eval writes them under
// <training folder>/<job>/identity_eval). ?step=N picks the round whose images
// and per-image scores are returned; default is the latest scored round.

async function readJson(p: string): Promise<any> {
  try {
    return JSON.parse(await fs.promises.readFile(p, 'utf-8'));
  } catch {
    return null;
  }
}

function slotImages(folder: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const modality of ['face', 'body', 'flex']) {
    const dir = path.join(folder, modality);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir)) {
      if (f.toLowerCase().endsWith('.png')) out[path.parse(f).name] = path.join(dir, f);
    }
  }
  return out;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const root = path.join(await getTrainingFolder(), job.name, 'identity_eval');
  const manifest = await readJson(path.join(root, 'manifest.json'));
  if (!manifest) {
    return NextResponse.json({ exists: false });
  }

  const rounds: any[] = [];
  try {
    const lines = (await fs.promises.readFile(path.join(root, 'results.jsonl'), 'utf-8')).split('\n');
    const byStep = new Map<number, any>();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const r = JSON.parse(line);
        byStep.set(r.step, r); // a resumed run may rescore a step; keep the latest
      } catch {}
    }
    rounds.push(...Array.from(byStep.values()).sort((a, b) => a.step - b.step));
  } catch {}

  const stepParam = request.nextUrl.searchParams.get('step');
  const step = stepParam !== null ? parseInt(stepParam) : rounds.length ? rounds[rounds.length - 1].step : null;
  let selected = null;
  if (step !== null && !isNaN(step)) {
    const folder = path.join(root, `step_${String(step).padStart(9, '0')}`);
    selected = {
      step,
      summary: await readJson(path.join(folder, 'checkpoint_summary.json')),
      per_image: await readJson(path.join(folder, 'per_image.json')),
      images: slotImages(folder),
    };
  }

  return NextResponse.json(
    {
      exists: true,
      calibration_id: manifest.calibration_id,
      metric_versions: manifest.metric_versions,
      gpus: manifest.gpus,
      created: manifest.created,
      slots: manifest.fingerprint?.slots ?? [],
      rounds,
      selected,
      baseline_images: slotImages(path.join(root, 'baseline')),
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
