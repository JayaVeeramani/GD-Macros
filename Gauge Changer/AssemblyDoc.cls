VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "AssemblyDoc"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Option Explicit

Private cAssy As IAssemblyDoc

Property Get GetUnderlyingObject() As IPartDoc
    GetUnderlyingObject = cAssy
End Property

Sub Initialize(swModelDoc As IModelDoc2)
    Set cAssy = swModelDoc
    
    cAssy.ResolveAllLightWeightComponents (False)
    
End Sub

Function GetUniquePartDoc() As Object

    Dim vComps As Variant
    vComps = cAssy.GetComponents(True)
    
    Set GetUniquePartDoc = CreateObject("Scripting.Dictionary")  'Path--> key,  part object -----> item
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As IComponent2
        Set swComp = vComps(i)
        
        If swComp.Visible = swComponentVisibilityState_e.swComponentVisible _
           And swComp.IsSuppressed = False Then
        
            Dim swModel As IModelDoc2
            Set swModel = swComp.GetModelDoc2
            
            Dim Path As String
            Path = swModel.GetPathName()
            
            If Not (GetUniquePartDoc.Exists(Path)) Then
                
                If swModel.GetType = swDocumentTypes_e.swDocPART Then
                
                    Dim oPart As PartDoc
                    Set oPart = NewPartDoc(swModel, swComp.ReferencedConfiguration)
                    
                    oPart.SetComponent = swComp
                    
                    GetUniquePartDoc.Add Path, oPart

                End If

            End If
            
        End If
    Next
    
End Function

Function GetPartNumber() As String

    Dim Path As String
    Path = cAssy.GetPathName()
    
    GetPartNumber = Mid(Path, InStrRev(Path, "\") + 1, InStrRev(Path, ".") - InStrRev(Path, "\") - 1)
    
End Function



