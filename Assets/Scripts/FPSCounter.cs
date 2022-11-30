using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FPSCounter : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
      Debug.Log("On Start of FPSCounter");
    }

    // Update is called once per frame
    void Update()
    {
      var fps = 1.0f / Time.deltaTime;
      // Debug.Log("Application.persistentDataPath: " + Application.persistentDataPath);
      Debug.Log("FPS: " + fps);
    }
}
