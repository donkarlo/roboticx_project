package com.noondreams.ndroutefinder;

import android.webkit.JavascriptInterface;

public final class MapBridge {
    private final MainActivity activity;
    public MapBridge(MainActivity activity){this.activity=activity;}
    @JavascriptInterface public void onMapClick(double latitude,double longitude){activity.runOnUiThread(()->activity.setStartPoint(latitude,longitude));}
}
