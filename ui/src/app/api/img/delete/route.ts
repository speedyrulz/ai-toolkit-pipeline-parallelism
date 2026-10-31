import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const fileExists = (p: string) => fs.promises.access(p).then(() => true).catch(() => false);
const isInside = (p: string, root: string) => p === root || p.startsWith(root + path.sep);

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetsPath = path.resolve(await getDatasetsRoot());
    const trainingPath = path.resolve(await getTrainingFolder());

    // resolve first: a raw prefix check lets `..` segments escape the allowed roots
    const imgPath = typeof body.imgPath === 'string' ? path.resolve(body.imgPath) : '';
    if (!imgPath || (!isInside(imgPath, datasetsPath) && !isInside(imgPath, trainingPath))) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|mp4|mp3|wav|flac|ogg)$/i.test(imgPath.toLowerCase())) {
      return NextResponse.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    if (!(await fileExists(imgPath))) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    await fs.promises.unlink(imgPath);

    // check for caption
    const captionPath = imgPath.replace(/\.[^/.\\]+$/, '') + '.txt';
    if (await fileExists(captionPath)) {
      // delete caption file
      await fs.promises.unlink(captionPath);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
