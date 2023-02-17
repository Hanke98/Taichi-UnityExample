using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Taichi;
using UnityEngine;
using System.Threading;
using Debug = UnityEngine.Debug;
using System;


public class QuantTest : MonoBehaviour
{   
    public AotModuleAsset Module;
    private ComputeGraph _ComputeGraph_init;
    private ComputeGraph _ComputeGraph_update;

    public NdArray<float> x;

    private int N = 10000;
    private int frame = 0;

    // Start is called before the first frame update
    void Start()
    {
        x = new NdArrayBuilder<float>().Shape(N).ElemShape(3).HostRead().HostWrite().Build();
        Application.targetFrameRate = 60;

        var cgraphs = Module.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        _ComputeGraph_init = cgraphs["init"];
        _ComputeGraph_update = cgraphs["update"];
        _ComputeGraph_init.LaunchAsync(new Dictionary<string, object>{});

    }


    // Update is called once per frame
    void Update()
    {
        frame += 1;
        var x_res = new float[x.Count];
        x.CopyToArray(x_res);
        var idx = (frame) % N;
        var d = BitConverter.GetBytes(x_res[idx * 3])
                            .Reverse()
                            .Select(x => Convert.ToString(x, 16))
                            // .Select(x => x.PadLeft(8, '0'))
                            .Aggregate("0x", (a, b) => a + "" + b);
        Debug.Log("frame: " + frame + " x: " + d);
        _ComputeGraph_update.LaunchAsync(new Dictionary<string, object>
        {
            { "x_arr", x }
        });
        Runtime.Submit();
    }
}
