'use client';

import { useState } from 'react';
import { Job } from '@prisma/client';
import usePollLoop from '@/hooks/usePollLoop';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl } from '@/utils/basic';

// Identity eval results (toolkit/identity_eval): the best checkpoint by the
// ranking gates, headline face / body / identity scores, the flex (prompt
// adherence) change from step 0, a per-round table with ranking, per-angle
// scores, baseline-vs-round images per slot with raw values against the
// anchors, each slot's images across rounds, and the reference set.

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
  flex?: { delta: number | null; valid_slots: number; slots: number };
  over_target_slots: number;
  warnings: string[];
  seconds: number;
}

interface SlotScore {
  modality: string;
  bin: string;
  status: string;
  reason: string | null;
  score?: number | null;
  max_ratio?: number | null;
  baseline?: 'paired' | 'pooled' | number | null;
  extra?: { yaw?: number; view?: string; det_score?: number } | null;
  features?: Record<string, { raw: number | null; score: number | null }>;
  delta?: number | null;
  confidence?: 'HIGH' | 'MEDIUM' | 'LOW' | null;
}

interface RankRow {
  step: number;
  identity: number | null;
  face: number | null;
  body: number | null;
  flex: number | null;
  eligible: boolean;
  reasons: string[];
  checkpoint: string | null;
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
  step_images?: Record<number, Record<string, string>>;
  ranking?: { criteria: { min_face: number; min_body: number; flex_max_drop: number }; rows: RankRow[] } | null;
  anchors?: {
    targets: Record<string, number | null>;
    baselines: Record<string, Record<string, number | null>>;
    reference_counts: Record<string, number>;
    reference_views: Record<string, Record<string, number>>;
    excluded_references: Record<string, { path: string; reason: string }[]>;
  } | null;
}

const fmt = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(1));
const signed = (v: number | null | undefined) =>
  v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);
const raw = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(3));
const fileName = (p: string) => p.split(/[\\/]/).pop() ?? p;
const BIN_ORDER = ['front', 'three_quarter', 'profile', 'side', 'back', 'any'];
const binSort = (a: string, b: string) =>
  (BIN_ORDER.indexOf(a) + 1 || 99) - (BIN_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b);

function BestCheckpoint({ ranking }: { ranking: IdentityData['ranking'] }) {
  if (!ranking?.rows?.length) return null;
  const c = ranking.criteria;
  const gates = `face ≥ ${c.min_face}, body ≥ ${c.min_body}, flex ≥ -${c.flex_max_drop}`;
  const best = ranking.rows[0];
  if (!best.eligible) {
    const top = ranking.rows.filter(r => r.identity !== null).sort((a, b) => b.identity! - a.identity!)[0];
    return (
      <div className="bg-gray-900 rounded-lg p-4 text-sm">
        <div className="text-xs uppercase text-gray-500">Best checkpoint</div>
        <div className="text-gray-300 mt-1">
          No round meets the ranking gates yet ({gates}).
          {top && (
            <span className="text-gray-400">
              {' '}
              Highest identity so far: step {top.step} ({fmt(top.identity)}; {top.reasons.join(', ')}).
            </span>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="bg-gray-900 rounded-lg p-4 text-sm">
      <div className="text-xs uppercase text-gray-500">Best checkpoint</div>
      <div className="text-gray-100 text-lg mt-1">
        step {best.step} · identity {fmt(best.identity)}
        <span className="text-gray-400 text-sm">
          {' '}
          (face {fmt(best.face)}, body {fmt(best.body)}, flex {signed(best.flex)})
        </span>
      </div>
      <div className={`text-xs mt-1 ${best.checkpoint ? 'text-gray-400' : 'text-yellow-500'}`}>
        {best.checkpoint
          ? fileName(best.checkpoint)
          : 'no checkpoint file saved at this step (not a save step, or removed by max saves)'}
      </div>
      <div className="text-xs text-gray-500 mt-1">gates: {gates}</div>
    </div>
  );
}

function Angles({ round }: { round: Round }) {
  const rows = (['face', 'body'] as const).map(m => [m, round[m]?.by_bin ?? {}] as const);
  const bins = Array.from(new Set(rows.flatMap(([, b]) => Object.keys(b)))).sort(binSort);
  if (!bins.length) return null;
  return (
    <table className="text-xs mb-4">
      <thead className="text-gray-500 text-left">
        <tr>
          <th className="pr-4 font-normal">angle</th>
          {bins.map(b => (
            <th key={b} className="pr-4 font-normal">
              {b.replace('_', ' ')}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([m, b]) => (
          <tr key={m}>
            <td className="pr-4 text-gray-500">{m}</td>
            {bins.map(bin => (
              <td key={bin} className="pr-4 text-gray-200">
                {fmt(b[bin])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

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

function FlexHeadline({ round }: { round: Round | null }) {
  const f = round?.flex;
  return (
    <div className="bg-gray-900 rounded-lg p-4 flex-1 min-w-[160px]">
      <div className="text-xs uppercase text-gray-500">Flex</div>
      <div className="text-3xl font-semibold text-gray-100">
        {signed(f?.delta)}
        <span className="text-base text-gray-500"> vs step 0</span>
      </div>
      <div className="text-xs text-gray-400 mt-1">
        {f ? `prompt adherence, ${f.valid_slots}/${f.slots} slots` : 'prompt adherence'}
      </div>
    </div>
  );
}

function Thumb({ path, label, small }: { path?: string; label: string; small?: boolean }) {
  const size = small ? 'w-16 h-16' : 'w-28 h-28';
  return (
    <div className="flex flex-col items-center shrink-0">
      {path ? (
        <a href={`/api/img/${encodeFilePathForUrl(path)}`} target="_blank" rel="noreferrer">
          <img
            src={`/api/img/${encodeFilePathForUrl(path)}`}
            alt={label}
            className={`${size} object-cover rounded bg-gray-800`}
            loading="lazy"
          />
        </a>
      ) : (
        <div className={`${size} rounded bg-gray-800 flex items-center justify-center text-xs text-gray-500`}>none</div>
      )}
      <div className="text-[10px] text-gray-500 mt-1">{label}</div>
    </div>
  );
}

export default function JobIdentity({ job }: { job: Job }) {
  const [data, setData] = useState<IdentityData | null>(null);
  const [step, setStep] = useState<number | null>(null);
  const [openSlot, setOpenSlot] = useState<string | null>(null);

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
  const rankOf = new Map<number, RankRow & { rank: number | null }>();
  let n = 0;
  for (const r of data.ranking?.rows ?? []) rankOf.set(r.step, { ...r, rank: r.eligible ? ++n : null });
  const stubRoles = Object.entries(data.metric_versions ?? {})
    .filter(([, v]) => v.includes('stub'))
    .map(([k]) => k);

  return (
    <div className="space-y-6">
      {stubRoles.length > 0 && (
        <p className="text-sm text-yellow-500">
          Placeholder scorer for {stubRoles.join(' and ')}: those numbers exercise the pipeline and do not measure
          identity (identity_eval backend is set to stub).
        </p>
      )}
      <BestCheckpoint ranking={data.ranking} />
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
        <FlexHeadline round={latest} />
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
                <th className="pr-4">Flex</th>
                <th className="pr-4">Coverage (face / body)</th>
                <th className="pr-4">Past target</th>
                <th className="pr-4">Rank</th>
                <th className="pr-4">Checkpoint</th>
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
                  <td className="pr-4">{signed(r.flex?.delta)}</td>
                  <td className="pr-4">
                    {pct(r.face.coverage)} / {pct(r.body.coverage)}
                  </td>
                  <td className="pr-4">{r.over_target_slots || ''}</td>
                  <td className="pr-4">
                    {rankOf.get(r.step)?.rank ? (
                      `#${rankOf.get(r.step)!.rank}`
                    ) : (
                      <span className="text-gray-500 text-xs">{rankOf.get(r.step)?.reasons.join(', ')}</span>
                    )}
                  </td>
                  <td className="pr-4">{rankOf.get(r.step)?.checkpoint ? 'saved' : ''}</td>
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
          {sel.summary && <Angles round={sel.summary} />}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {(data.slots ?? []).map(s => {
              const r = perImage[s.id];
              const open = openSlot === s.id;
              return (
                <div key={s.id} className="border border-gray-800 rounded p-2">
                  <div className="flex gap-3 items-start">
                    <Thumb path={data.baseline_images?.[s.id]} label="step 0" />
                    <Thumb path={sel.images[s.id]} label={`step ${sel.step}`} />
                    <div className="text-xs text-gray-400 min-w-0">
                      <div className="text-gray-200">
                        {s.id} · {s.bin} · {s.renderer === 'helper' ? 'second gpu' : 'trainer gpu'}
                      </div>
                      <div className="truncate" title={s.prompt}>
                        {s.prompt}
                      </div>
                      {s.modality === 'flex' && r ? (
                        <div className="mt-1">
                          adherence <span className="text-gray-100">{signed(r.delta)}</span> vs step 0
                          {r.status !== 'VALID' && (
                            <span className="text-yellow-500">
                              {' '}
                              · {r.status}
                              {r.reason ? `: ${r.reason}` : ''}
                            </span>
                          )}
                        </div>
                      ) : r ? (
                        <div className="mt-1">
                          score <span className="text-gray-100">{fmt(r.score)}</span>
                          {r.features && (
                            <span className="text-gray-500">
                              {' ('}
                              {Object.entries(r.features)
                                .map(([k, v]) => `${k.replace(/^(face|body)_/, '')} ${fmt(v.score)}`)
                                .join(', ')}
                              {')'}
                            </span>
                          )}
                          {r.extra?.view && (
                            <span className="text-gray-500">
                              {' '}
                              · seen as {r.extra.view} ({Math.round(r.extra.yaw ?? 0)}°)
                            </span>
                          )}
                          {r.confidence && (
                            <span className={r.confidence === 'LOW' ? 'text-yellow-500' : 'text-gray-500'}>
                              {' '}
                              · {r.confidence.toLowerCase()} confidence
                            </span>
                          )}
                          {r.baseline === 'pooled' && <span className="text-yellow-500"> · median baseline</span>}
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
                      <button
                        className="text-gray-500 hover:text-gray-300 mt-1"
                        onClick={() => setOpenSlot(open ? null : s.id)}
                      >
                        {open ? 'hide details' : 'details'}
                      </button>
                    </div>
                  </div>
                  {open && (
                    <div className="mt-2 text-xs text-gray-400 space-y-2">
                      {r?.features && (
                        <table>
                          <thead className="text-gray-500 text-left">
                            <tr>
                              <th className="pr-3 font-normal">feature</th>
                              <th className="pr-3 font-normal">raw</th>
                              <th className="pr-3 font-normal">step 0</th>
                              <th className="pr-3 font-normal">target</th>
                              <th className="pr-3 font-normal">score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(r.features).map(([k, v]) => (
                              <tr key={k}>
                                <td className="pr-3">{k}</td>
                                <td className="pr-3 text-gray-200">{raw(v.raw)}</td>
                                <td className="pr-3">{raw(data.anchors?.baselines?.[s.id]?.[k])}</td>
                                <td className="pr-3">{raw(data.anchors?.targets?.[k])}</td>
                                <td className="pr-3 text-gray-200">{fmt(v.score)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                      <div className="flex gap-2 overflow-x-auto pb-1">
                        <Thumb small path={data.baseline_images?.[s.id]} label="0" />
                        {rounds.map(rd => (
                          <Thumb
                            key={rd.step}
                            small
                            path={data.step_images?.[rd.step]?.[s.id]}
                            label={String(rd.step)}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {data.anchors?.reference_counts && (
        <div className="bg-gray-900 rounded-lg p-4 text-xs text-gray-400">
          <div className="uppercase text-gray-500 mb-2">References</div>
          {Object.entries(data.anchors.reference_counts).map(([m, count]) => {
            const views = Object.entries(data.anchors?.reference_views?.[m] ?? {})
              .sort(([a], [b]) => binSort(a, b))
              .map(([v, c]) => `${c} ${v.replace('_', ' ')}`)
              .join(', ');
            const excluded = data.anchors?.excluded_references?.[m] ?? [];
            return (
              <div key={m} className="mb-1">
                <span className="text-gray-200">{m}</span>: {count} used{views ? ` (${views})` : ''}
                {excluded.length > 0 && (
                  <span title={excluded.map(e => `${fileName(e.path)}: ${e.reason}`).join('\n')}>
                    {' '}
                    · {excluded.length} excluded (hover for reasons)
                  </span>
                )}
              </div>
            );
          })}
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
