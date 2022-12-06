using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// using LuaFramework;
using UnityEngine.UI;

public class Logger : MonoBehaviour
{
    public bool showFlag = true;
    public int logCount = 100;
    private string m_ShowLog = string.Empty;
    private Queue<string> logQueue = new Queue<string>();
    // Start is called before the first frame update
    void Start()
    {
        Application.logMessageReceived += WriteUnityLog;
    }
    void WriteUnityLog(string log, string stackTrace, LogType type)
    {
        if (!showFlag) return;
        WriteInLogQueue(log);
    }

    void WriteInLogQueue(string log)
    {
        logQueue.Enqueue(log);
        while (logQueue.Count > logCount)
        {
            logQueue.Dequeue();
        }
        m_ShowLog = string.Empty;
        foreach (string onelog in logQueue)
        {
            m_ShowLog = m_ShowLog + "\r\n" + onelog;
        }
    }
    void OnGUI()
    {
        if (showFlag)
        {
            GUIStyle style = new GUIStyle();
            style.fontSize = 30;
            style.normal.textColor = Color.red;

            //GUI.color = Color.red;
            GUI.Label(new Rect(40, 200, 1000, 1000), m_ShowLog, style);
        }
    }
}
