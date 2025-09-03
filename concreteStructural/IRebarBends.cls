VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IRebarBends"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cBody As SldWorks.Body2
Private cComp As SldWorks.Component2
Private cCompModel As SldWorks.ModelDoc2
Private cToroidalFace As SldWorks.Face2
Private cView As SldWorks.View


Private cxCenter As Double
Private cyCenter As Double
Private czCenter As Double

Private cxCG As Double
Private cyCG As Double
Private czCG As Double

Private cCenterPoint(2) As Double
Private cCGPoint(2) As Double

Private orgViewPosition As Variant
Private orgScaleDecimal As Double

Private cIsBalloonRight As Boolean
Private cIsBalloonTop As Boolean

Property Get IsBalloonRight() As Boolean

    IsBalloonRight = cIsBalloonRight
    
End Property

Property Get IsBalloonTop() As Boolean

    IsBalloonTop = cIsBalloonTop
    
End Property

Sub Initialize(swComp As SldWorks.Component2, swView As SldWorks.View) ' oCutlist As ICutlist,
    
    Set cComp = swComp
    Set cCompModel = ResolveAndGetModelDoc(swComp)
    Set cBody = swComp.GetBody()
    Set cView = swView
    cIsBalloonRight = False
    cIsBalloonTop = False
    
    Set cToroidalFace = GetToroidalFace()
    
    If Not cToroidalFace Is Nothing Then
    
        Call GetTorusCenters
        Call GetCGPoint
        Call UpdateSheetMaxMinDimensions(swView)
        Call GetBallPosBoolean
        
    End If
    
End Sub

Sub GetBallPosBoolean()
    
    If cxCenter > cxCG Then
    
        cIsBalloonRight = True
    
    End If
    
    If cyCenter > cyCG Then
    
        cIsBalloonTop = True
        
    End If

End Sub

Sub GetTorusCenters()
    
    Dim swSurface As SldWorks.Surface
    Set swSurface = cToroidalFace.GetSurface
    
    Dim vTorusParams As Variant
    vTorusParams = swSurface.TorusParams
    
    cCenterPoint(0) = vTorusParams(0)
    cCenterPoint(1) = vTorusParams(1)
    cCenterPoint(2) = vTorusParams(2)
    
End Sub

Sub GetCGPoint()
    
    Dim swModelExt As SldWorks.ModelDocExtension
    Set swModelExt = cCompModel.Extension
    
    Dim swMassProp As SldWorks.MassProperty
    Set swMassProp = swModelExt.CreateMassProperty
    
    Dim vCG As Variant
    vCG = swMassProp.CenterOfMass
    
    cCGPoint(0) = vCG(0)
    cCGPoint(1) = vCG(1)
    cCGPoint(2) = vCG(2)
  
End Sub

Function GetToroidalFace() As SldWorks.Face2

    Dim vFaces  As Variant
    vFaces = cBody.GetFaces
    
    Dim i As Integer
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        Dim swSurface As SldWorks.Surface
        Set swSurface = swFace.GetSurface
        
        If swSurface.IsTorus Then
        
            Set GetToroidalFace = swFace
            Exit Function
            
        End If
    
    Next i
    
End Function

Private Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View)

    Dim CenterPoint As Variant
    CenterPoint = GetComponentPointInSheetSpace(cComp, cCenterPoint, swView)

    Dim CGPoint As Variant
    CGPoint = GetComponentPointInSheetSpace(cComp, cCGPoint, swView)
    
    cxCG = CGPoint(0)
    cyCG = CGPoint(1)
    czCG = CGPoint(2)
    
    cxCenter = CenterPoint(0)
    cyCenter = CenterPoint(1)
    czCenter = CenterPoint(2)

    Debug.Print cxCG
    Debug.Print cComp.Name2
    Debug.Print cyCG
    
    Debug.Print cxCenter
    Debug.Print cyCenter
    
End Sub

Sub CheckForUpdateInMaxMinDimensions(swView As SldWorks.View)

    If IsViewChanged(swView) Then
        
        Call UpdateSheetMaxMinDimensions(swView)
        Set cView = swView
        orgScaleDecimal = swView.ScaleDecimal
        orgViewPosition = swView.Position
            
    End If

End Sub

Private Function IsViewChanged(swView As SldWorks.View) As Boolean
    
    IsViewChanged = False

    If (swView.Name = cView.Name) Then

        Dim actualViewPosition As Variant
        actualViewPosition = swView.Position
    
        If Not (swView.ScaleDecimal = orgScaleDecimal) Or Not (orgViewPosition(0) = actualViewPosition(0)) _
                Or Not (orgViewPosition(1) = actualViewPosition(1)) Or Not (swView.Angle = cView.Angle) Then

            IsViewChanged = True
     
        End If
            
    Else
            
        IsViewChanged = True
            
    End If
    
End Function

