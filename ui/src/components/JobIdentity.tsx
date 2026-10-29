'use client';

import { useState } from 'react';
import { Job } from '@prisma/client';
import usePollLoop from '@/hooks/usePollLoop';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl } from '@/utils/basic';

// Identity eval results (toolkit/identity_eval): headline face / body /
// identity scores, a per-round table, and baseline-vs-round images per slot.

interface Modality {
  score: number | null;
  coverage: number;
  by_bin: Record<string, number | null>;
  status: string;
}

interface Round {
  step: number;
  face: Modality;
  body: Modality;
  identity: { score: number | null; valid: boolean };
  over_target_slots: number;
  warnings: string[];
  seconds: number;
}

interface SlotScore {
  modality: string;
  bin: string;
  status: string;
  reason: string | null;
  score: number | null;
  max_ratio?: number | null;
}

interface IdentityData {
  exists: boolean;
  calibration_id?: string;
  metric_versions?: Record<string, string>;
  gpus?: Record<string, number>;
  created?: string;
  slots?: { id: string; modality: string; bin: string; prompt: string; renderer: string }[];
  rounds?: Round[];
  selected?: {
    step: number;
    summary: Round | null;
    per_image: Record<string, SlotScore> | null;
    images: Record<string, string>;
  } | null;
  baseline_images?: Record<string, string>;
}

const fmt = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(1));
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);

function Headline({ label, value, sub }: { label: string; value: number | null | undefined; sub: string }) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 flex-1 min-w-[160px]">
      <div className="text-xs uppercase text-gray-500">{label}</div>
      <div className="text-3xl font-semibold text-gray-100">
        {fmt(value)}
        <span className="text-base text-gray-500"> / 100</span>
      </div>
      <div className="text-xs text-gray-400 mt-1">{sub}</div>
    </div>
  );
}

function Thumb({ path, label }: { path?: string; label: string }) {
  return (
    <div className="flex flex-col items-center">
      {path ? (
        <img
          src={`/api/img/${encodeFilePathForUrl(path)}`}
          alt={label}
          className="w-28 h-28 object-cover rounded bg-gray-800"
          loading="lazy"
        />
      ) : (
        <div className="w-28 h-28 rounded bg-gray-800 flex items-center justify-center text-xs text-gray-500">none</div>
      )}
      <div className="text-[10px] text-gray-500 mt-1">{label}</div>
    </div>
  );
}

export default function JobIdentity({ job }: { job: Job }) {
  const [data, setData] = useState<IdentityData | null>(null);
  const [step, setStep] = useState<number | null>(null);

  usePollLoop(
    () =>
      apiClient
        .get(`/api/jobs/${job.id}/identity${step !== null ? `?step=${step}` : ''}`)
        .then(res => setData(res.data))
        .catch(() => {}),
    10000,
    [job.id, step],
  );

  if (data === null) return <p className="text-gray-400">Loading...</p>;
  if (!data.exists) {
    return (
      <p className="text-gray-400">
        Identity eval has not run for this job. Turn on the Identity Eval card in the job config to score sample rounds
        for face and body likeness.
      </p>
    );
  }

  const rounds = data.rounds ?? [];
  const latest = rounds.length ? rounds[rounds.length - 1] : null;
  const sel = data.selected;
  const perImage = sel?.per_image ?? {};
  const stub = Object.values(data.metric_versions ?? {}).some(v => v.includes('stub'));

  return (
    <div className="space-y-6">
      {stub && (
        <p className="text-sm text-yellow-500">
          Placeholder scorer (metric version {Object.values(data.metric_versions ?? {}).join(', ')}): these numbers
          exercise the pipeline and do not measure identity yet.
        </p>
      )}
      <div className="flex flex-wrap gap-4">
        <Headline label="Face" value={latest?.face.score} sub={`coverage ${pct(latest?.face.coverage)}`} />
        <Headline label="Body" value={latest?.body.score} sub={`coverage ${pct(latest?.body.coverage)}`} />
        <Headline
          label="Identity"
          value={latest?.identity.score}
          sub={
            latest ? (latest.identity.valid ? `step ${latest.step}` : 'N/A: a modality is missing') : 'no rounds yet'
          }
        />
      </div>
      {latest?.warnings?.length ? (
        <ul className="text-sm text-yellow-500 list-disc pl-5">
          {latest.warnings.map(w => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      ) : null}

      <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
        <div className="text-xs uppercase text-gray-500 mb-2">Rounds</div>
        {rounds.length === 0 ? (
          <p className="text-sm text-gray-400">Calibrated; waiting for the first eval round.</p>
        ) : (
          <table className="text-sm w-full">
            <thead className="text-gray-500 text-left">
              <tr>
                <th className="pr-4">Step</th>
                <th className="pr-4">Face</th>
                <th className="pr-4">Body</th>
                <th className="pr-4">Identity</th>
                <th className="pr-4">Coverage (face / body)</th>
                <th className="pr-4">Past target</th>
                <th className="pr-4">Time</th>
              </tr>
            </thead>
            <tbody>
              {rounds.map(r => (
                <tr
                  key={r.step}
                  onClick={() => setStep(r.step)}
                  className={`cursor-pointer hover:bg-gray-800 ${sel?.step === r.step ? 'bg-gray-800' : ''}`}
                >
                  <td className="pr-4">{r.step}</td>
                  <td className="pr-4">{fmt(r.face.score)}</td>
                  <td className="pr-4">{fmt(r.body.score)}</td>
                  <td className="pr-4">{fmt(r.identity.score)}</td>
                  <td className="pr-4">
                    {pct(r.face.coverage)} / {pct(r.body.coverage)}
                  </td>
                  <td className="pr-4">{r.over_target_slots || ''}</td>
                  <td className="pr-4">{Math.round(r.seconds)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {sel && (
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-xs uppercase text-gray-500 mb-3">Step {sel.step}: baseline vs this round</div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {(data.slots ?? []).map(s => {
              const r = perImage[s.id];
              return (
                <div key={s.id} className="flex gap-3 items-start border border-gray-800 rounded p-2">
                  <Thumb path={data.baseline_images?.[s.id]} label="step 0" />
                  <Thumb path={sel.images[s.id]} label={`step ${sel.step}`} />
                  <div className="text-xs text-gray-400 min-w-0">
                    <div className="text-gray-200">
                      {s.id} · {s.bin} · {s.renderer === 'helper' ? 'second gpu' : 'trainer gpu'}
                    </div>
                    <div className="truncate" title={s.prompt}>
                      {s.prompt}
                    </div>
                    {s.modality === 'flex' ? (
                      <div className="mt-1 text-gray-500">flexibility scoring arrives in Phase 1</div>
                    ) : r ? (
                      <div className="mt-1">
                        score <span className="text-gray-100">{fmt(r.score)}</span>
                        {r.max_ratio !== undefined && r.max_ratio !== null && r.max_ratio > 1 && (
                          <span className="text-yellow-500"> · {Math.round(r.max_ratio * 100)}% of target</span>
                        )}
                        {r.status !== 'VALID' && (
                          <span className="text-yellow-500">
                            {' '}
                            · {r.status}
                            {r.reason ? `: ${r.reason}` : ''}
                          </span>
                        )}
                      </div>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="text-xs text-gray-500">
        calibration {data.calibration_id?.slice(0, 19)} · created {data.created} · scoring gpus{' '}
        {Object.entries(data.gpus ?? {})
          .map(([k, v]) => `${k}: ${v < 0 ? 'cpu' : `#${v}`}`)
          .join(', ')}
      </div>
    </div>
  );
}
