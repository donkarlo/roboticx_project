package com.noondreams.ndroutefinder;

import java.util.List;

public final class TrackMath {
    private static final double EARTH_RADIUS_M=6371000.0;
    public double distanceKm(GpxTrack track){List<GpxPoint> p=track.getPoints();double total=0;for(int i=1;i<p.size();i++)total+=distanceMeters(p.get(i-1),p.get(i));return total/1000.0;}
    public double maxSlopePercent(GpxTrack track){List<GpxPoint> p=track.getPoints();if(p.size()<2)return Double.POSITIVE_INFINITY;double max=0;for(int i=0;i<p.size();i++){if(p.get(i).elevation==null)continue;double accumulated=0;int j=i+1;while(j<p.size()&&accumulated<60){accumulated+=distanceMeters(p.get(j-1),p.get(j));j++;}if(j-1>=p.size()||accumulated<20)continue;GpxPoint end=p.get(j-1);if(end.elevation==null)continue;double slope=Math.abs(end.elevation-p.get(i).elevation)/accumulated*100.0;if(slope>max)max=slope;}return max;}
    public double distanceMeters(GpxPoint a,GpxPoint b){double lat1=Math.toRadians(a.latitude),lat2=Math.toRadians(b.latitude),dLat=lat2-lat1,dLon=Math.toRadians(b.longitude-a.longitude);double s1=Math.sin(dLat/2),s2=Math.sin(dLon/2);double h=s1*s1+Math.cos(lat1)*Math.cos(lat2)*s2*s2;return 2*EARTH_RADIUS_M*Math.asin(Math.sqrt(h));}
}
