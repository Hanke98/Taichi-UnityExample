using System.Collections.Generic;
using System.Linq;
using Taichi;
using UnityEngine;
using System.Diagnostics;
using System.Threading;
using Debug = UnityEngine.Debug;

public partial class FractalQuant : MonoBehaviour {
    const int N = 4;
    const int WIDTH = 640 * N;
    const int HEIGHT = 320 * N;

    public AotModuleAsset Module;
    private Kernel _Kernel_fractal;
    private ComputeGraph _ComputeGraph_fractal;
    private ComputeGraph _ComputeGraph_get_data;
    private NdArray<float> _Canvas;

    int frame = 0;
    long numTicks = 0;
    private MeshRenderer _MeshRenderer;
    private Texture2D _Texture;
    private Color[] _FractalDataColor;

    // Start is called before the first frame update
    void Start() {
        var kernels = Module.GetAllKernels().ToDictionary(x => x.Name);
        var cgraphs = Module.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (kernels.ContainsKey("fractal")) {
            _Kernel_fractal = kernels["fractal"];
        }
        if (cgraphs.ContainsKey("fractal")) {
            _ComputeGraph_fractal = cgraphs["fractal"];
        }
        if (cgraphs.ContainsKey("get_data")) {
          _ComputeGraph_get_data = cgraphs["get_data"];
        }

        _Canvas = new NdArray<float>(new int[] { WIDTH, HEIGHT }, true, false);
        _Texture = new Texture2D(WIDTH, HEIGHT);

        _FractalDataColor = new Color[WIDTH * HEIGHT];

        _MeshRenderer = GetComponent<MeshRenderer>();
        _MeshRenderer.material.mainTexture = _Texture;
    }

    Color GetColor() {
        var iframe = Time.frameCount;

        float pos = iframe % 100;
        float alpha = (pos < 50) ? (pos / 50.0f) : (2.0f - pos / 50.0f);

        var color = new Color();
        color.r = alpha * 0.75f;
        color.g = 0.75f;
        color.b = (1.0f - alpha) * 0.75f;
        color.a = 1.0f;
        return color;
    }

    // Update is called once per frame
    void Update() {
        // Note that we are reading data from last frame here.
        var fractal = new float[WIDTH * HEIGHT];
        _Canvas.CopyToArray(fractal);
        for (int i = 0; i < _FractalDataColor.Length; ++i) {
            var v = fractal[i];
            _FractalDataColor[i] = new Color(v, v, v);
        }
        _Texture.SetPixels(_FractalDataColor);
        _Texture.Apply();
        _MeshRenderer.material.color = GetColor();

        // Now launch kernels and compute graphs, but it won't be
        // immediately executed on graphics device.
        float t = Time.frameCount * 0.03f;
        frame += 1;

        Stopwatch sw = new Stopwatch();
        sw.Start();
        for (int i =0; i < 4; ++i ) {
          if (_ComputeGraph_fractal != null) {
              _ComputeGraph_fractal.LaunchAsync(new Dictionary<string, object> {});
          }
          if (_Kernel_fractal != null) {
              _Kernel_fractal.LaunchAsync(t, _Canvas);
          }

          if (_ComputeGraph_get_data != null) {
              _ComputeGraph_get_data.LaunchAsync(new Dictionary<string, object> {
                { "cvs", _Canvas },
              });
          }
        }

        // Everything settled. Submit launched kernels and compute graphs to
        // the device for execution. Note that we can only submit ONCE in a
        // frame and we CANNOT wait on the device, because `Update is called
        // on the GAME THREAD yet everything for rendering is created in the
        // RENDER THREAD. `TaichiRuntime.Submit` will submit the commands in
        // the RENDER THREAD.
        Runtime.Submit();
        // Thread.Sleep(1000);
        sw.Stop();

        // Debug.Log("frame: "+ frame);
        if (frame < 50) return;
        if (frame > 100) return;

        long nanosecPerTick = (1000L*1000L*1000L) / Stopwatch.Frequency;
        numTicks += sw.ElapsedTicks;
        var nanosec = (numTicks * nanosecPerTick) / (frame-50);
        var musec = nanosec / 1000;
        Debug.Log(string.Format("Total {0} mus", musec));
    }
}
