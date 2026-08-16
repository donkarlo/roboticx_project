package com.noondreams.ndroutefinder;

import android.content.ContentResolver;
import android.content.Context;
import android.net.Uri;
import androidx.documentfile.provider.DocumentFile;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public final class FileTreeReader {
    public List<GpxTrack> readGpxRecursively(Context context,Uri treeUri){List<GpxTrack> tracks=new ArrayList<>();DocumentFile root=DocumentFile.fromTreeUri(context,treeUri);if(root!=null)readDirectory(context.getContentResolver(),root,tracks);return tracks;}
    private void readDirectory(ContentResolver resolver,DocumentFile file,List<GpxTrack> tracks){if(file.isDirectory()){for(DocumentFile child:file.listFiles())readDirectory(resolver,child,tracks);return;}String name=file.getName();if(name==null||!name.toLowerCase().endsWith(".gpx"))return;try(InputStream in=resolver.openInputStream(file.getUri())){if(in!=null){GpxTrack track=new GpxParser().parse(in);if(!track.getPoints().isEmpty())tracks.add(track);}}catch(Exception ignored){}}
}
