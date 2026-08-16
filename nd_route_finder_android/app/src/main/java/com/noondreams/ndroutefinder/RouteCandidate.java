package com.noondreams.ndroutefinder;

public final class RouteCandidate {
    public final GpxTrack track;
    public final double distanceKm;
    public final double maxSlopePercent;
    public final String rawGpx;
    public RouteCandidate(GpxTrack track, double distanceKm, double maxSlopePercent, String rawGpx) {
        this.track = track;
        this.distanceKm = distanceKm;
        this.maxSlopePercent = maxSlopePercent;
        this.rawGpx = rawGpx;
    }
}
