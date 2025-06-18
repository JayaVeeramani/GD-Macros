VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "PartDoc"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

Private cPart As IPartDoc
Private cModel As IModelDoc2
Private cComp As IComponent2

Private cPath As String

Private cConfig As IConfiguration
Private cConfigName As String

Private cConfigPropMgr As SldWorks.CustomPropertyManager
Private cCustomPropMgr As SldWorks.CustomPropertyManager

Private cProfile As String

Private wasResolved As Boolean
Private Val As String

Property Get GetComponent() As SldWorks.Component2
    Set GetComponent = cComp
End Property


Property Let SetComponent(swComp As SldWorks.Component2)
    Set cComp = swComp
End Property

Property Get ConfigName() As String
    ConfigName = cConfigName
End Property

Property Get Path() As String
    Path = cPath
End Property

Property Get GetIpartDocObject() As IPartDoc
    Set GetUnderlyingObject = cPart
End Property

Property Get GetModelDocObject() As IModelDoc2
    Set GetModelDocObject = cModel
End Property

Property Get GetConfig() As IConfiguration
     Set GetConfig = cConfig
End Property

Property Get Profile() As String
    Profile = cProfile
End Property

Sub Initialize(swModel As SldWorks.ModelDoc2, ConfigName As String)
    
    Set cModel = swModel
    Set cPart = cModel

    cPath = cModel.GetPathName
    
    cConfigName = ConfigName
    
    Set cConfig = cModel.GetConfigurationByName(cConfigName)
    Set cConfigPropMgr = cConfig.CustomPropertyManager
    Set cCustomPropMgr = cModel.Extension.CustomPropertyManager("")
    cProfile = GetPropVal("Profile")


End Sub

Function IsSheetMetal() As Boolean

    IsSheetMetal = False
    
    Dim vBodies As Variant
    vBodies = cPart.GetBodies2(swBodyType_e.swSolidBody, True)
    
    Dim swBody As IBody2
    
    If Not IsEmpty(vBodies) Then
        Set swBody = vBodies(0)
        If swBody.IsSheetMetal Then
            IsSheetMetal = True
        End If
    End If

End Function

Function GetPartNumber() As String
    
    GetPartNumber = Mid(cPath, InStrRev(cPath, "\") + 1, InStrRev(cPath, ".") - InStrRev(cPath, "\") - 1)
    
End Function


Public Function GetPropVal(PropName As String) As String

    Dim PropVal As String
    cConfigPropMgr.Get5 PropName, False, Val, PropVal, wasResolved
    
    If PropVal = "" Then
    
        cCustomPropMgr.Get5 PropName, False, Val, PropVal, wasResolved
        
    End If
    
    GetPropVal = PropVal
    
End Function

Public Function SetPropVal(PropName As String, PropVal As String) As String

    cCustomPropMgr.Set2 PropName, PropVal

End Function










