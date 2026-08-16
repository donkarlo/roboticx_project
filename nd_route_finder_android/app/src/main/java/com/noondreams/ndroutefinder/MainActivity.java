package com.noondreams.ndroutefinder;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import org.json.JSONObject;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int REQUEST_GPX_FOLDER=1001,REQUEST_SAVE_GPX=1002;
    private EditText latitudeInput,longitudeInput,distanceInput,slopeInput;private TextView folderText,statusText;private Spinner activitySpinner;private Button generateButton,saveButton;private WebView mapView;private Uri gpxTreeUri;private String pendingGpx;private final List<GpxTrack> previousTracks=new ArrayList<>();private final MapSerializer mapSerializer=new MapSerializer();
    @Override protected void onCreate(Bundle b){super.onCreate(b);setTitle("nd_route_finder");buildUi();configureMap();}
    private void buildUi(){LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setPadding(12,10,12,10);ScrollView formScroll=new ScrollView(this);LinearLayout form=new LinearLayout(this);form.setOrientation(LinearLayout.VERTICAL);form.setPadding(0,0,0,8);LinearLayout folderRow=row();folderText=new TextView(this);folderText.setText("GPX root folder: not selected");folderText.setSingleLine(true);Button folderButton=new Button(this);folderButton.setText("Browse GPX folder");folderButton.setOnClickListener(v->chooseFolder());folderRow.addView(folderText,new LinearLayout.LayoutParams(0,LinearLayout.LayoutParams.WRAP_CONTENT,1f));folderRow.addView(folderButton);form.addView(folderRow);activitySpinner=new Spinner(this);activitySpinner.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"Cycling","Hiking"}));form.addView(labeled("Activity",activitySpinner));latitudeInput=decimalInput("47.070700");longitudeInput=decimalInput("15.439500");distanceInput=decimalInput("20.0");slopeInput=decimalInput("20.0");form.addView(labeled("Start latitude",latitudeInput));form.addView(labeled("Start longitude",longitudeInput));form.addView(labeled("Target distance (km)",distanceInput));form.addView(labeled("Maximum slope (%)",slopeInput));generateButton=new Button(this);generateButton.setText("Generate route");generateButton.setOnClickListener(v->generateRoute());form.addView(generateButton);saveButton=new Button(this);saveButton.setText("Save generated GPX");saveButton.setEnabled(false);saveButton.setOnClickListener(v->chooseOutputFile());form.addView(saveButton);statusText=new TextView(this);statusText.setText("Tap the map to choose the start point.");statusText.setPadding(4,8,4,8);form.addView(statusText);formScroll.addView(form);root.addView(formScroll,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,LinearLayout.LayoutParams.WRAP_CONTENT));mapView=new WebView(this);root.addView(mapView,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,0,1f));setContentView(root);}
    private LinearLayout row(){LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);return r;}
    private View labeled(String label,View control){LinearLayout r=row();TextView t=new TextView(this);t.setText(label);t.setPadding(2,12,8,4);r.addView(t,new LinearLayout.LayoutParams(0,LinearLayout.LayoutParams.WRAP_CONTENT,.42f));r.addView(control,new LinearLayout.LayoutParams(0,LinearLayout.LayoutParams.WRAP_CONTENT,.58f));return r;}
    private EditText decimalInput(String value){EditText i=new EditText(this);i.setSingleLine(true);i.setInputType(android.text.InputType.TYPE_CLASS_NUMBER|android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL|android.text.InputType.TYPE_NUMBER_FLAG_SIGNED);i.setText(value);return i;}
    private void configureMap(){WebSettings s=mapView.getSettings();s.setJavaScriptEnabled(true);s.setDomStorageEnabled(true);s.setUserAgentString(s.getUserAgentString()+" nd_route_finder_android/1.0");mapView.addJavascriptInterface(new MapBridge(this),"Android");mapView.setWebViewClient(new WebViewClient(){@Override public void onPageFinished(WebView v,String url){super.onPageFinished(v,url);try{setStartPoint(Double.parseDouble(latitudeInput.getText().toString()),Double.parseDouble(longitudeInput.getText().toString()));refreshPreviousTracks();}catch(NumberFormatException ignored){}}});mapView.loadUrl("file:///android_asset/map.html");}
    public void setStartPoint(double lat,double lon){latitudeInput.setText(String.format(Locale.US,"%.6f",lat));longitudeInput.setText(String.format(Locale.US,"%.6f",lon));mapView.evaluateJavascript(String.format(Locale.US,"setStart(%.8f,%.8f);",lat,lon),null);statusText.setText(String.format(Locale.US,"Start selected: %.6f, %.6f",lat,lon));}
    private void chooseFolder(){Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION|Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION|Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);startActivityForResult(i,REQUEST_GPX_FOLDER);}
    private void chooseOutputFile(){if(pendingGpx==null)return;Intent i=new Intent(Intent.ACTION_CREATE_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("application/gpx+xml");i.putExtra(Intent.EXTRA_TITLE,"generated_round_trip.gpx");startActivityForResult(i,REQUEST_SAVE_GPX);}
    @Override protected void onActivityResult(int req,int result,Intent data){super.onActivityResult(req,result,data);if(result!=RESULT_OK||data==null||data.getData()==null)return;Uri uri=data.getData();if(req==REQUEST_GPX_FOLDER){getContentResolver().takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION);gpxTreeUri=uri;folderText.setText("GPX root: "+uri);loadPreviousTracks();}else if(req==REQUEST_SAVE_GPX)saveGpx(uri);}
    private void loadPreviousTracks(){if(gpxTreeUri==null)return;statusText.setText("Loading previous GPX tracks ...");new Thread(()->{List<GpxTrack> loaded=new FileTreeReader().readGpxRecursively(this,gpxTreeUri);runOnUiThread(()->{previousTracks.clear();previousTracks.addAll(loaded);refreshPreviousTracks();statusText.setText("Loaded "+loaded.size()+" previous GPX track(s).");});}).start();}
    private void refreshPreviousTracks(){String json=mapSerializer.tracksToJson(previousTracks);mapView.evaluateJavascript("setPreviousTracks("+JSONObject.quote(json)+");",null);}
    private void generateRoute(){final double lat,lon,distance,maxSlope;try{lat=Double.parseDouble(latitudeInput.getText().toString());lon=Double.parseDouble(longitudeInput.getText().toString());distance=Double.parseDouble(distanceInput.getText().toString());maxSlope=Double.parseDouble(slopeInput.getText().toString());}catch(NumberFormatException e){Toast.makeText(this,"Check latitude, longitude, distance and slope.",Toast.LENGTH_LONG).show();return;}if(distance<=0||maxSlope<=0){Toast.makeText(this,"Distance and maximum slope must be positive.",Toast.LENGTH_LONG).show();return;}boolean cycling=activitySpinner.getSelectedItemPosition()==0;setBusy(true);statusText.setText("Generating route ...");new Thread(()->{try{RouteCandidate r=new RouteGenerator().generate(lat,lon,distance,maxSlope,cycling,m->runOnUiThread(()->statusText.setText(m)));runOnUiThread(()->{pendingGpx=r.rawGpx;saveButton.setEnabled(true);String json=mapSerializer.trackToJson(r.track);mapView.evaluateJavascript("setGeneratedTrack("+JSONObject.quote(json)+");",null);statusText.setText(String.format(Locale.US,"Done — %.2f km, max slope %.1f%%. Tap Save generated GPX.",r.distanceKm,r.maxSlopePercent));setBusy(false);});}catch(Exception e){runOnUiThread(()->{statusText.setText("Failed: "+e.getMessage());Toast.makeText(this,"Route generation failed: "+e.getMessage(),Toast.LENGTH_LONG).show();setBusy(false);});}}).start();}
    private void setBusy(boolean b){generateButton.setEnabled(!b);activitySpinner.setEnabled(!b);latitudeInput.setEnabled(!b);longitudeInput.setEnabled(!b);distanceInput.setEnabled(!b);slopeInput.setEnabled(!b);}
    private void saveGpx(Uri uri){if(pendingGpx==null)return;try(OutputStream out=getContentResolver().openOutputStream(uri,"wt")){if(out==null)throw new IllegalStateException("Could not open output file.");byte[] bytes=pendingGpx.getBytes(StandardCharsets.UTF_8);out.write(bytes);out.flush();statusText.setText("GPX saved: "+uri+" ("+bytes.length+" bytes)");}catch(Exception e){statusText.setText("Save failed: "+e.getMessage());Toast.makeText(this,"Save failed: "+e.getMessage(),Toast.LENGTH_LONG).show();}}
    @Override public void onBackPressed(){if(mapView.canGoBack())mapView.goBack();else super.onBackPressed();}
    @Override protected void onDestroy(){if(mapView!=null)mapView.destroy();super.onDestroy();}
}
