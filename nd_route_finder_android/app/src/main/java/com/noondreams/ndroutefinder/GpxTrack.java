package com.noondreams.ndroutefinder;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class GpxTrack {
    private final List<GpxPoint> points;
    public GpxTrack(List<GpxPoint> points) { this.points = Collections.unmodifiableList(new ArrayList<>(points)); }
    public List<GpxPoint> getPoints() { return points; }
}
