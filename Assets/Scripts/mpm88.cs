using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Taichi;
using UnityEngine.Rendering;
using System.Linq;
using UnityEngine.UIElements;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

public class mpm88 : MonoBehaviour
{
    const int WIDTH = 240;
    const int HEIGHT = 240;

    public AotModuleAsset mpm88Module;

    public NdArray<float> pos;

    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_update;

    private NdArray<float> _Canvas;
    private MeshRenderer _MeshRenderer;
    private Texture2D _Texture;
    private Texture2D _Clear;
    private Color[] _mpm88Color;
    private float r = 2.0f;
    private Color ParticleColor=new Color(1.0f,0.0f,0.0f);
    int frame = 0;
    long numTicks = 0;
    double delta_time = 0.0;

    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 60;
        var cgraphs = mpm88Module.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if(cgraphs.Count>0)
        {
            _Compute_Graph_g_init = cgraphs["init"];
            _Compute_Graph_g_update = cgraphs["update"];
        }

        _Canvas = new NdArray<float>(new int[] { WIDTH, HEIGHT }, true, false);
        _Texture = new Texture2D(WIDTH, HEIGHT);
        _Clear = new Texture2D(WIDTH, HEIGHT);
        _mpm88Color = new Color[WIDTH* HEIGHT];
        _MeshRenderer = GetComponent<MeshRenderer>();
        _MeshRenderer.material.mainTexture = _Texture;

        int n_particles = 2000;//Do not exceed 20000 to ensure smooth running of the demo

        //Taichi Allocate memory,hostwrite are not considered
        pos = new NdArrayBuilder<float>().Shape(n_particles).ElemShape(3).HostRead().Build();

        if(_Compute_Graph_g_init !=null)
        {
            _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object>{});
        }
        else
        {
            //kernel initialize
        }
        // Runtime.Submit();
    }

    void in_circle_or_not(ref Color[] mpm88_color,ref float[] pos)
    {
        for(int i = 0;i<mpm88_color.Length;++i)
        {
           for(int j = 0;j<pos.Length;j+=3)
           {
                Vector2 tt = new Vector2(pos[j ] * WIDTH, pos[j+1]*HEIGHT);
                
                Vector2 pixel = new Vector2(i % WIDTH, i / WIDTH);
                Vector2 distance_vecor = tt- pixel; 
                if(distance_vecor.magnitude<=r)
                {
                    mpm88_color[i] = ParticleColor;
                    break;
                }
           }
            if (mpm88_color[i] == ParticleColor) continue;
            else mpm88_color[i] = Color.white;       
        }
    }
    void in_circle_or_notv2(ref Color[] mpm88_color, ref float[] pos)
    {
        for (int j = 0; j < pos.Length; j += 3)
        {

            Vector2 tt = new Vector2((int)(pos[j] * WIDTH),(int)( pos[j + 1] * HEIGHT));
            var idx = (int)(tt.y*WIDTH+tt.x);
			idx = 0;
			// var idx = 0;
            mpm88_color[idx] = ParticleColor;
        }
    }

    // Update is called once per frame
    void Update()
    {
        frame += 1;
        if (frame >= 80) return;
        float[] temp2 = new float[pos.Count];
        pos.CopyToArray(temp2);

        _mpm88Color = new Color[WIDTH * HEIGHT];
        in_circle_or_notv2(ref _mpm88Color, ref temp2);

        _Texture.SetPixels(_mpm88Color);
        _Texture.Apply();
        var sw = new Stopwatch();
        sw.Start();
        if (_Compute_Graph_g_update!=null)
        {
            _Compute_Graph_g_update.LaunchAsync(new Dictionary<string, object>
            {
                {"x_arr", pos}
            });
        }
        Runtime.Submit();
        sw.Stop();
        if (frame <= 20) return;
        if (frame > 80) return;
        delta_time += Time.deltaTime;
        
        var fps = 1.0f / delta_time * (frame -20);
        // long nanosecPerTick = (1000L*1000L*1000L) / Stopwatch.Frequency;
        // numTicks += sw.ElapsedTicks;
        // var nanosec = (numTicks * nanosecPerTick) / (frame-20);
        // var musec = nanosec / 1000;
        // Debug.Log(string.Format("Total {0} mus", musec));
        Debug.Log("fps: " + fps.ToString());
    }
}
