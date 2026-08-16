package com.noondreams.ndroutefinder;

import java.util.List;

public final class MapSerializer {
    public String tracksToJson(List<GpxTrack> tracks) {
        StringBuilder builder = new StringBuilder("[");
        for (int index = 0; index < tracks.size(); index++) {
            if (index > 0) {
                builder.append(',');
            }
            appendTrack(builder, tracks.get(index));
        }
        return builder.append(']').toString();
    }

    public String trackToJson(GpxTrack track) {
        StringBuilder builder = new StringBuilder();
        appendTrack(builder, track);
        return builder.toString();
    }

    private void appendTrack(StringBuilder builder, GpxTrack track) {
        builder.append('[');
        List<GpxPoint> points = track.getPoints();
        for (int index = 0; index < points.size(); index++) {
            if (index > 0) {
                builder.append(',');
            }
            GpxPoint point = points.get(index);
            builder.append('[')
                    .append(point.latitude)
                    .append(',')
                    .append(point.longitude)
                    .append(']');
        }
        builder.append(']');
    }
}
