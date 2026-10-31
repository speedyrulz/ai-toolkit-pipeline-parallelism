'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, Box, File, Folder, FolderOpen, HardDrive, Home, Loader2 } from 'lucide-react';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';

interface DirEntry {
  name: string;
  path: string;
  isModel: boolean;
}

interface FileEntry {
  name: string;
  path: string;
  size: number;
}

interface Listing {
  path: string;
  parent: string | null;
  isModel: boolean;
  dirs: DirEntry[];
  files: FileEntry[];
  roots: string[];
  modelsPath: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onPick: (path: string) => void;
  // current field value; the browser opens at it when it is a local path
  initialPath?: string | null;
}

const fmtSize = (n: number) => (n > 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`);

// hub repo ids ("org/name") are not local paths; don't try to open them
const looksLocal = (p?: string | null) => !!p && (/^[a-zA-Z]:[\\/]/.test(p) || p.startsWith('/') || p.startsWith('.'));

// Browse button plus its modal, for dropping into an input's suffix slot.
export function ModelBrowseButton({
  value,
  onPick,
  className = 'flex items-center gap-1 text-gray-300 hover:text-white',
}: {
  value?: string | null;
  onPick: (path: string) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" title="Browse for a local model" onClick={() => setOpen(true)} className={className}>
        <FolderOpen className="w-4 h-4" /> Browse
      </button>
      <ModelBrowserModal isOpen={open} onClose={() => setOpen(false)} initialPath={value} onPick={onPick} />
    </>
  );
}

// styled to sit beside a select as its own control
export const standaloneBrowseClass =
  'shrink-0 flex items-center gap-1 px-2 text-sm text-gray-300 hover:text-white bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm';

export default function ModelBrowserModal({ isOpen, onClose, onPick, initialPath }: Props) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState('');
  // requests can overlap (Enter in the path box is not disabled while loading);
  // only the latest one may update the listing
  const requestSeq = useRef(0);

  const load = useCallback((p?: string) => {
    const mine = ++requestSeq.current;
    setLoading(true);
    setError(null);
    return apiClient
      .get('/api/models/browse', { params: p ? { path: p } : {} })
      .then(r => {
        if (mine !== requestSeq.current) return;
        setListing(r.data);
        setPathInput(r.data.path);
      })
      .catch(e => {
        if (mine !== requestSeq.current) return;
        setError(e?.response?.data?.error || e?.message || 'Unable to read folder');
      })
      .finally(() => {
        if (mine === requestSeq.current) setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    load(looksLocal(initialPath) ? initialPath! : undefined);
  }, [isOpen]);

  const pick = (p: string) => {
    onPick(p);
    onClose();
  };

  const rowClass = 'w-full flex items-center gap-2 px-2 py-1 rounded text-xs text-left hover:bg-gray-800';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Browse for Model" size="lg">
      <div className="text-gray-200">
        <div className="flex items-center gap-1 mb-2">
          <button
            type="button"
            title="Up one folder"
            disabled={!listing?.parent || loading}
            onClick={() => listing?.parent && load(listing.parent)}
            className="p-1.5 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-40"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
          <button
            type="button"
            title="Models folder"
            disabled={loading}
            onClick={() => load(listing?.modelsPath)}
            className="p-1.5 rounded bg-gray-800 hover:bg-gray-700 disabled:opacity-40"
          >
            <Home className="w-4 h-4" />
          </button>
          <input
            value={pathInput}
            onChange={e => setPathInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                e.preventDefault();
                load(pathInput);
              }
            }}
            className="flex-1 min-w-0 px-2 py-1 text-xs font-mono bg-gray-950 border border-gray-700 rounded text-gray-100"
          />
        </div>
        {listing && listing.roots.length > 1 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {listing.roots.map(r => (
              <button
                type="button"
                key={r}
                onClick={() => load(r)}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300"
              >
                <HardDrive className="w-3 h-3" /> {r}
              </button>
            ))}
          </div>
        )}
        {error && <div className="text-xs text-red-400 mb-2">{error}</div>}
        <div className="h-[50vh] overflow-y-auto pr-1 rounded-md bg-gray-950 border border-gray-800 p-1">
          {loading && (
            <div className="text-gray-500 text-sm flex items-center gap-2 p-2">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading…
            </div>
          )}
          {!loading && listing && (
            <div className="space-y-0.5">
              {listing.dirs.length === 0 && listing.files.length === 0 && (
                <div className="text-xs text-gray-500 p-2">No folders or model files here.</div>
              )}
              {listing.dirs.map(d => (
                <div key={d.path} className="flex items-center gap-1">
                  <button type="button" onClick={() => load(d.path)} className={rowClass} title={d.path}>
                    {d.isModel ? (
                      <Box className="w-4 h-4 text-blue-400 shrink-0" />
                    ) : (
                      <Folder className="w-4 h-4 text-amber-400 shrink-0" />
                    )}
                    <span className="truncate flex-1 font-mono">{d.name}</span>
                  </button>
                  {d.isModel && (
                    <button
                      type="button"
                      onClick={() => pick(d.path)}
                      className="shrink-0 px-2 py-0.5 rounded text-[11px] bg-blue-700 hover:bg-blue-600 text-white"
                    >
                      Select
                    </button>
                  )}
                </div>
              ))}
              {listing.files.map(f => (
                <button
                  type="button"
                  key={f.path}
                  onClick={() => pick(f.path)}
                  className={`${rowClass} hover:bg-blue-900/40`}
                  title={f.path}
                >
                  <File className="w-4 h-4 text-gray-400 shrink-0" />
                  <span className="truncate flex-1 font-mono">{f.name}</span>
                  <span className="text-gray-500 shrink-0">{fmtSize(f.size)}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="mt-3 flex items-center justify-between gap-2">
          <span className="text-[11px] text-gray-500">
            Click a model file to select it, or open a folder and use it as a model folder.
          </span>
          <button
            type="button"
            disabled={!listing || loading}
            onClick={() => listing && pick(listing.path)}
            className={`shrink-0 flex items-center gap-1 px-3 py-1 rounded text-xs disabled:opacity-40 ${
              listing?.isModel ? 'bg-blue-700 hover:bg-blue-600 text-white' : 'bg-gray-800 hover:bg-gray-700'
            }`}
          >
            <FolderOpen className="w-3.5 h-3.5" /> Use this folder
          </button>
        </div>
      </div>
    </Modal>
  );
}
