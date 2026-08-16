package com.noondreams.ndroutefinder;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public final class RouteGenerator {
    private static final String ENDPOINT="https://brouter.de/brouter-web/brouter";
    private static final double EARTH_RADIUS_M=6371000.0;
    private final GpxParser parser=new GpxParser();
    private final TrackMath math=new TrackMath();
    public RouteCandidate generate(double startLat,double startLon,double targetKm,double maxSlopePercent,boolean cycling,ProgressCallback callback)throws Exception{
        RouteCandidate best=null;double[] radiusFactors={0.18,0.22,0.26,0.30};int attempt=0;
        for(double radiusFactor:radiusFactors){for(int bearing=0;bearing<360;bearing+=45){attempt++;callback.onProgress("Candidate "+attempt+": routing ...");double radiusKm=targetKm*radiusFactor;double[] p1=destination(startLat,startLon,radiusKm,bearing);double[] p2=destination(startLat,startLon,radiusKm,bearing+120);double[] p3=destination(startLat,startLon,radiusKm,bearing+240);String gpx=requestRoute(startLat,startLon,p1,p2,p3,cycling);GpxTrack track=parser.parse(gpx);if(track.getPoints().size()<2)continue;double distance=math.distanceKm(track);double slope=math.maxSlopePercent(track);callback.onProgress(String.format(Locale.US,"Candidate %d: %.2f km, max slope %.1f%%",attempt,distance,slope));RouteCandidate candidate=new RouteCandidate(track,distance,slope,gpx);if(best==null||score(candidate,targetKm,maxSlopePercent)<score(best,targetKm,maxSlopePercent))best=candidate;if(slope<=maxSlopePercent&&Math.abs(distance-targetKm)<=targetKm*0.15)return candidate;}}
        if(best==null)throw new IllegalStateException("No route candidate could be generated.");if(best.maxSlopePercent>maxSlopePercent)throw new IllegalStateException(String.format(Locale.US,"No route stayed below %.1f%%. Best: %.1f%%, %.2f km.",maxSlopePercent,best.maxSlopePercent,best.distanceKm));return best;
    }
    private double score(RouteCandidate c,double target,double maxSlope){double d=Math.abs(c.distanceKm-target)/Math.max(target,.1);double s=c.maxSlopePercent>maxSlope?10+(c.maxSlopePercent-maxSlope)/Math.max(maxSlope,1):0;return d+s;}
    private String requestRoute(double startLat,double startLon,double[] p1,double[] p2,double[] p3,boolean cycling)throws Exception{String lonLats=point(startLon,startLat)+"|"+point(p1[1],p1[0])+"|"+point(p2[1],p2[0])+"|"+point(p3[1],p3[0])+"|"+point(startLon,startLat);String profile=cycling?"trekking":"hiking-beta";String query="lonlats="+URLEncoder.encode(lonLats,StandardCharsets.UTF_8.name())+"&profile="+URLEncoder.encode(profile,StandardCharsets.UTF_8.name())+"&alternativeidx=0&format=gpx";URL url=new URL(ENDPOINT+"?"+query);HttpURLConnection c=(HttpURLConnection)url.openConnection();c.setConnectTimeout(20000);c.setReadTimeout(60000);c.setRequestProperty("User-Agent","nd_route_finder_android/1.0");int code=c.getResponseCode();if(code!=200)throw new IllegalStateException("Routing service HTTP "+code);StringBuilder text=new StringBuilder();try(BufferedReader r=new BufferedReader(new InputStreamReader(c.getInputStream(),StandardCharsets.UTF_8))){String line;while((line=r.readLine())!=null)text.append(line).append('\n');}finally{c.disconnect();}return text.toString();}
    private String point(double lon,double lat){return String.format(Locale.US,"%.6f,%.6f",lon,lat);}
    private double[] destination(double lat,double lon,double distanceKm,double bearingDeg){double angular=distanceKm*1000/EARTH_RADIUS_M,bearing=Math.toRadians(bearingDeg),lat1=Math.toRadians(lat),lon1=Math.toRadians(lon);double lat2=Math.asin(Math.sin(lat1)*Math.cos(angular)+Math.cos(lat1)*Math.sin(angular)*Math.cos(bearing));double lon2=lon1+Math.atan2(Math.sin(bearing)*Math.sin(angular)*Math.cos(lat1),Math.cos(angular)-Math.sin(lat1)*Math.sin(lat2));return new double[]{Math.toDegrees(lat2),Math.toDegrees(lon2)};}
    public interface ProgressCallback{void onProgress(String message);}
}
