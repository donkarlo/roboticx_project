package com.noondreams.ndroutefinder;

import android.util.Xml;
import org.xmlpull.v1.XmlPullParser;
import java.io.InputStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

public final class GpxParser {
    public GpxTrack parse(InputStream inputStream) throws Exception { XmlPullParser p=Xml.newPullParser(); p.setInput(inputStream,null); return parse(p); }
    public GpxTrack parse(String xml) throws Exception { XmlPullParser p=Xml.newPullParser(); p.setInput(new StringReader(xml)); return parse(p); }
    private GpxTrack parse(XmlPullParser parser) throws Exception {
        List<GpxPoint> points=new ArrayList<>(); Double lat=null,lon=null,ele=null; boolean in=false; int event=parser.getEventType();
        while(event!=XmlPullParser.END_DOCUMENT){
            if(event==XmlPullParser.START_TAG){String name=parser.getName(); if("trkpt".equals(name)||"rtept".equals(name)){lat=Double.parseDouble(parser.getAttributeValue(null,"lat"));lon=Double.parseDouble(parser.getAttributeValue(null,"lon"));ele=null;in=true;} else if(in&&"ele".equals(name)){ele=Double.parseDouble(parser.nextText().trim());}}
            else if(event==XmlPullParser.END_TAG){String name=parser.getName(); if(in&&("trkpt".equals(name)||"rtept".equals(name))){if(lat!=null&&lon!=null)points.add(new GpxPoint(lat,lon,ele));in=false;}}
            event=parser.next();
        }
        return new GpxTrack(points);
    }
}
