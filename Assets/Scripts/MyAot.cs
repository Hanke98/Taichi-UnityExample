using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using Taichi;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Profiling;
using System.Threading;
using Debug=UnityEngine.Debug;

public class MyAot : MonoBehaviour
{
    public AotModuleAsset MyAotModule; 
    private ComputeGraph _Compute_Graph_g_init;
    private ComputeGraph _Compute_Graph_g_assign;

    public NdArray<float> x;
    public float[] x_array;
    public float[] y_array;
    public NdArray<float> y;
    public NdArray<float> r;
    private const int dim = 2;
    private const int N = 20000;
    private int iters = 0;
    private long numTicks = 0;

    // Start is called before the first frame update
    void Start()
    {
      // x = new NdArray<float>(N);
      // y = new NdArray<float>(N);
      // r = new NdArray<float>(N);
      x = new NdArrayBuilder<float>().Shape(N).ElemShape(dim).HostWrite().HostRead().Build();
      y = new NdArrayBuilder<float>().Shape(N).ElemShape(dim).HostWrite().HostRead().Build();
      r = new NdArrayBuilder<float>().Shape(N).ElemShape(dim).HostWrite().HostRead().Build();

      x_array = new float[N*dim];
      y_array = new float[N*dim];
      var cgraphs = MyAotModule.GetAllComputeGrpahs().ToDictionary(x => x.Name);
      if (cgraphs.Count > 0) {
        _Compute_Graph_g_init = cgraphs["init"];
        _Compute_Graph_g_assign = cgraphs["assign"];
      }
      // if (_Compute_Graph_g_init != null) {
      //   _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object> {
      //       { "x", x },
      //       { "y", y },
      //       { "r", r },
      //   });
      // }
      // if (_Compute_Graph_g_assign != null) {
      //   _Compute_Graph_g_assign.LaunchAsync(new Dictionary<string, object>());
      // }
      // Runtime.Submit();
    }

    // Update is called once per frame
    void Update()
    {
        iters += 1;
        if(iters > 150) return;
        if (iters < 100) return;

        Stopwatch sw = new Stopwatch();
        sw.Start();
        _Compute_Graph_g_assign.LaunchAsync(new Dictionary<string, object>());
        Runtime.Submit();
        // Thread.Sleep(300);
        // _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object> {
        //     { "x", x },
        //     { "y", y },
        //     { "r", r },
        // });
        // Runtime.Submit();
        // Thread.Sleep(300);
        sw.Stop();
        // UnityEngine.Debug.Log(string.Format("Total {0} ms", sw.ElapsedMilliseconds));

        // Thread.Sleep(1000);
        x.CopyToArray(x_array);
        y.CopyToArray(y_array);
        long nanosecPerTick = (1000L*1000L*1000L) / Stopwatch.Frequency;
        numTicks += sw.ElapsedTicks;
        var nanosec = (numTicks * nanosecPerTick) / (iters-100);
        var musec = nanosec / 1000;
        Debug.Log(string.Format("Total {0} mus", musec));
        // for(int i = 0; i < N; ++i) {
        //   if (y_array[i] > -0.49f) {
        //     UnityEngine.Debug.Log(string.Format("{0} th y_array is not -0.5", i));
        //   }
        // }
        this.transform.position = new Vector3(x_array[N-1], y_array[N-1], 0.0f);
    }
}
