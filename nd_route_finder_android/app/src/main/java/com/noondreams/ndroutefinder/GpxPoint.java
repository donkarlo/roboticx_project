package com.noondreams.ndroutefinder;

public final class GpxPoint {
    public final double latitude;
    public final double longitude;
    public final Double elevation;
    public GpxPoint(double latitude, double longitude, Double elevation) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.elevation = elevation;
    }
}
