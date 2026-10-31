import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

const isInside = (p: string, root: string) => p === root || p.startsWith(root + path.sep);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { caption, ext } = body;
    const datasetsPath = path.resolve(await getDatasetsRoot());
    // resolve first: a raw prefix check lets `..` segments escape the datasets root
    const imgPath = typeof body.imgPath === 'string' ? path.resolve(body.imgPath) : '';
    if (!imgPath || !isInside(imgPath, datasetsPath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    try {
      await fs.promises.access(imgPath);
    } catch {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    // check for caption (default extension txt); the extension is caller supplied
    const captionExt = ((ext || 'txt') as string).replace(/^\.+/, '').trim() || 'txt';
    if (!/^[a-z0-9]{1,10}$/i.test(captionExt)) {
      return NextResponse.json({ error: 'Invalid caption extension' }, { status: 400 });
    }
    const captionPath = imgPath.replace(/\.[^/.\\]+$/, '') + '.' + captionExt;
    // save caption to file
    await fs.promises.writeFile(captionPath, caption);

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
