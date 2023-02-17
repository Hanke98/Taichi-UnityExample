using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Taichi;
using UnityEngine;
using System.Threading;

using Debug = UnityEngine.Debug;

public class Pbf2d : MonoBehaviour
{
    
    const int WIDTH = 800;
    const int HEIGHT = 800;
    private const float screen_to_world_ratio = 10.0f / 80.0f;
    private const int num_particles = 3000;

    public AotModuleAsset Module;
    private ComputeGraph _ComputeGraph_init;
    private ComputeGraph _ComputeGraph_update;

    private NdArray<float> positions;
    private NdArray<float> board_states;

    private MeshRenderer _MeshRenderer;
    private Texture2D _Texture;
    private Color[] _Pbf2dDataColor;

    private long numTicks = 0;
    private int frame = 0;
	private float total_time = 0.0f;
    
    // Start is called before the first frame update
    void Start()
    {
        Application.targetFrameRate = 60;
        var cgraphs = Module.GetAllComputeGrpahs().ToDictionary(x => x.Name);
        if (cgraphs.ContainsKey("init"))
            _ComputeGraph_init = cgraphs["init"];
        if (cgraphs.ContainsKey("update"))
            _ComputeGraph_update = cgraphs["update"];

        _Texture = new Texture2D(WIDTH, HEIGHT);

        _Pbf2dDataColor = new Color[WIDTH * HEIGHT];

        _MeshRenderer = GetComponent<MeshRenderer>();
        _MeshRenderer.material.mainTexture = _Texture;

        positions = new NdArrayBuilder<float>().Shape(num_particles).ElemShape(2).HostRead().HostWrite().Build();
        board_states = new NdArrayBuilder<float>().Shape().ElemShape(2).HostRead().HostWrite().Build();
        if (_ComputeGraph_init != null)
        {
            _ComputeGraph_init.LaunchAsync(new Dictionary<string, object>
            {
                { "board_states", board_states }
            });
        }

        Debug.Log("Start success!");
    }

    // Update is called once per frame
    void Update()
    {
		// return;
		if (frame > 100) return;
        var positions_2d = new float[positions.Count];
        positions.CopyToArray(positions_2d);
        for (int i = 0; i < _Pbf2dDataColor.Length; ++i)
        {
            _Pbf2dDataColor[i] = new Color(1, 1, 1);
        }
        
        _Texture.SetPixels(_Pbf2dDataColor);
        for (int i = 0; i < positions_2d.Length; i += 2)
        {
            var x = positions_2d[i] * screen_to_world_ratio;
            var y = positions_2d[i + 1] * screen_to_world_ratio;
            _Texture.SetPixel((int)x, (int)y, new Color(0, 0, 0));
        }
        
        var board_bound = new float[board_states.Count];
        board_states.CopyToArray(board_bound);
        int bound_x = (int)(board_bound[0] * screen_to_world_ratio);
        for (int i = 0; i < HEIGHT; ++i)
        {
            _Texture.SetPixel(bound_x, i, Color.black);
            _Texture.SetPixel(bound_x + 1, i, Color.black);
        }
        
        _Texture.Apply();

        // return;
        // float time_delta = Time.deltaTime;
        float time_delta = 1.0f/120.0f;
        var sw = new Stopwatch();
        sw.Start();
        if (_ComputeGraph_update != null)
        {
            _ComputeGraph_update.LaunchAsync(new Dictionary<string, object>
            {
                { "positions_nda", positions}
            });
        }
        sw.Stop();
        Runtime.Submit();
        frame += 1;
        if (frame < 20) return; 
        if (frame > 100) return; 
        // var fps = 1.0f / Time.deltaTime;
		total_time += Time.deltaTime;
        var fps = 1.0f / total_time * (frame -20);
        // long nanosecPerTick = (1000L*1000L*1000L) / Stopwatch.Frequency;
        // numTicks += sw.ElapsedTicks;
        // var nanosec = (numTicks * nanosecPerTick) / (frame-20);
        // var musec = nanosec / 1000;
        // Debug.Log(string.Format("Total {0} mus", musec));
        Debug.Log("fps: " + fps.ToString());
    }
}
