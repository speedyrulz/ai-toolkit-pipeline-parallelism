import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

// Identity eval results for a job (toolkit/identity_eval writes them under
// <training folder>/<job>/identity_eval). ?step=N picks the round whose images
// and per-image scores are returned; default is the latest scored round. The
// checkpoint ranking (ranking.json) is joined with the checkpoint files still
// on disk: <job>_<step 9 digits>.safetensors, or <job>.safetensors for the end.

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

// step -> checkpoint path; the unnumbered final save is returned as step -1
function checkpoints(jobFolder: string, jobName: string): Map<number, string> {
  const out = new Map<number, string>();
  if (!fs.existsSync(jobFolder)) return out;
  for (const f of fs.readdirSync(jobFolder)) {
    if (!f.startsWith(jobName) || !f.endsWith('.safetensors')) continue;
    const rest = f.slice(jobName.length, -'.safetensors'.length);
    const m = rest.match(/^_(\d{9})$/);
    if (m) out.set(parseInt(m[1]), path.join(jobFolder, f));
    else if (rest === '') out.set(-1, path.join(jobFolder, f));
  }
  return out;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const jobFolder = path.join(await getTrainingFolder(), job.name);
  const root = path.join(jobFolder, 'identity_eval');
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

  // ranking rows with the checkpoint saved at that step; the final save
  // belongs to the last scored round when no numbered file has that step
  const ranking = await readJson(path.join(root, 'ranking.json'));
  if (ranking?.rows) {
    const ckpts = checkpoints(jobFolder, job.name);
    const lastStep = rounds.length ? rounds[rounds.length - 1].step : null;
    for (const row of ranking.rows) {
      row.checkpoint = ckpts.get(row.step) ?? (row.step === lastStep ? ckpts.get(-1) : undefined) ?? null;
    }
  }

  // every round's images, for the per-slot history strip
  const stepImages: Record<number, Record<string, string>> = {};
  for (const r of rounds) {
    stepImages[r.step] = slotImages(path.join(root, `step_${String(r.step).padStart(9, '0')}`));
  }

  const anchors = await readJson(path.join(root, 'anchors.json'));

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
      step_images: stepImages,
      ranking,
      status: await readJson(path.join(root, 'status.json')),
      anchors: anchors
        ? {
            targets: anchors.targets,
            baselines: anchors.baselines,
            reference_counts: anchors.reference_counts,
            reference_views: anchors.reference_views,
            excluded_references: anchors.excluded_references,
          }
        : null,
    },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
