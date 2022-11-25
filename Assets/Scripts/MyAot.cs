using System.Collections.Generic;
using System.IO;
using System.Linq;
using Taichi;
using UnityEngine;
using UnityEngine.Rendering;


public class MyAot : MonoBehaviour
{
    public AotModuleAsset MyAotModule; 
    private ComputeGraph _Compute_Graph_g_init;

    public NdArray<float> x;
    public float[] x_array;
    public float[] y_array;
    public NdArray<float> y;
    public NdArray<float> r;
    private const int dim = 2;
    private const int N = 200;

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
      }
      if (_Compute_Graph_g_init != null) {
        _Compute_Graph_g_init.LaunchAsync(new Dictionary<string, object> {
            { "x", x },
            { "y", y },
            { "r", r },
        });
      }
      // Runtime.Submit();
    }

    // Update is called once per frame
    void Update()
    {
        x.CopyToArray(x_array);
        y.CopyToArray(y_array);
        this.transform.position = new Vector3(x_array[0], y_array[0], 0.0f);
    }
}
