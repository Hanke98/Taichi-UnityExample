using System.Collections.Generic;
using System.Linq;
using Taichi;
using UnityEngine;
using System.Threading;

public class Pbf2d : MonoBehaviour
{
    const int WIDTH = 800;
    const int HEIGHT = 400;
    private const float screen_to_world_ratio = 10.0f;
    private const int num_particles = 1200;

    public AotModuleAsset Module;
    private ComputeGraph _ComputeGraph_init;
    private ComputeGraph _ComputeGraph_update;

    private NdArray<float> positions;
    private NdArray<float> board_states;

    private MeshRenderer _MeshRenderer;
    private Texture2D _Texture;
    private Color[] _Pbf2dDataColor;

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
                { "positions", positions },
                { "board_states", board_states }
            });
        }

        Debug.Log("Start success!");
    }

    // Update is called once per frame
    void Update()
    {
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

        float time_delta = Time.deltaTime;
        if (_ComputeGraph_update != null)
        {
          for (int i = 0; i < 2; ++i)
            _ComputeGraph_update.LaunchAsync(new Dictionary<string, object>
            {
                { "positions", positions },
                { "board_states", board_states },
                { "time_delta", time_delta }
            });
        }

        Runtime.Submit();
        var fps = 1.0f / Time.deltaTime;
        Debug.Log("fps: " + fps.ToString());
    }
}
