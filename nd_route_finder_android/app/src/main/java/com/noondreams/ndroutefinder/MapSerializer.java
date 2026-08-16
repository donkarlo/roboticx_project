package com.noondreams.ndroutefinder;

import org.json.JSONArray;
import java.util.List;

public final class MapSerializer {
    public String tracksToJson(List<GpxTrack> tracks){JSONArray all=new JSONArray();for(GpxTrack track:tracks){JSONArray points=new JSONArray();for(GpxPoint point:track.getPoints()){JSONArray p=new JSONArray();p.put(point.latitude);p.put(point.longitude);points.put(p);}all.put(points);}return all.toString();}
    public String trackToJson(GpxTrack track){JSONArray points=new JSONArray();for(GpxPoint point:track.getPoints()){JSONArray p=new JSONArray();p.put(point.latitude);p.put(point.longitude);points.put(p);}return points.toString();}
}
