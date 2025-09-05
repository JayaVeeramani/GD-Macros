VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "IRebarBody"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = False
Attribute VB_Exposed = False
Private cBody As SldWorks.Body2
Private cComp As SldWorks.Component2
Private cxMin As Double
Private cyMin As Double
Private cZMin As Double

Private cxMax As Double
Private cyMax As Double
Private cZMax As Double

Private cMid As Double


Private cxMinSkPt As Double
Private cyMinSkPt As Double
Private cZMinSkPt As Double

Private cxMaxSkPt As Double
Private cyMaxSkPt As Double
Private cZMaxSkPt As Double

Private cIsHorizontal As Boolean

Private cView As SldWorks.View
Private cBodyMinPoint(2) As Double
Private cBodyMaxPoint(2) As Double

'Private cCutlist As ICutlist

Private cLeftEdge As SldWorks.Edge
Private cRightEdge As SldWorks.Edge
Private cBottomEdge As SldWorks.Edge
Private cTopEdge As SldWorks.Edge

Private orgViewPosition As Variant
Private orgScaleDecimal As Double

Private cStructuralFeat As SldWorks.Feature
Private cSketchSegment As SldWorks.SketchSegment


Property Get SketchSegment() As SldWorks.SketchSegment

    Set SketchSegment = cSketchSegment
    
End Property

Property Get IsHorizontal() As Boolean

    IsHorizontal = cIsHorizontal
    
End Property

Property Get LeftEdge() As SldWorks.Edge

    Set LeftEdge = cLeftEdge

End Property

Property Get RightEdge() As SldWorks.Edge

    Set RightEdge = cRightEdge
    
End Property

Property Get TopEdge() As SldWorks.Edge

    Set TopEdge = cTopEdge

End Property

Property Get BottomEdge() As SldWorks.Edge

    Set BottomEdge = cBottomEdge

End Property

Property Get GetComponent() As SldWorks.Component2

    Set GetComponent = cComp
    
End Property


Property Get GetBody() As SldWorks.Body2

    Set GetBody = cBody
    
End Property

Property Get Mid() As Double
 
    Mid = cMid
    
End Property

Property Get xMinSketchPoint() As Double

    xMinSketchPoint = cxMinSkPt

End Property

Property Get xMaxSketchPoint() As Double

    xMaxSketchPoint = cxMaxSkPt

End Property

Property Get yMinSketchPoint() As Double

    yMinSketchPoint = cyMinSkPt

End Property

Property Get yMaxSketchPoint() As Double

    yMaxSketchPoint = cyMaxSkPt

End Property

Property Get xMin() As Double
 
    xMin = cxMin
    
End Property

Property Get yMin() As Double

    yMin = cyMin
    
End Property

Property Get zMin() As Double

    zMin = cZMin
    
End Property

Property Get xMax() As Double

    xMax = cxMax
    
End Property

Property Get yMax() As Double

    yMax = cyMax
    
End Property

Property Get zMax() As Double

    zMax = cZMax
    
End Property

Sub Initialize(swComp As SldWorks.Component2, swBody As SldWorks.Body2, swView As SldWorks.View) ' oCutlist As ICutlist,
    
    Set cComp = swComp
    Set cBody = swBody
    Set cView = swView
    
    orgViewPosition = cView.Position
    orgScaleDecimal = cView.ScaleDecimal

    Dim vBodyBounds As Variant
    vBodyBounds = swBody.GetBodyBox
        
    cBodyMinPoint(0) = vBodyBounds(0)
    cBodyMinPoint(1) = vBodyBounds(1)
    cBodyMinPoint(2) = vBodyBounds(2)
        
    cBodyMaxPoint(0) = vBodyBounds(3)
    cBodyMaxPoint(1) = vBodyBounds(4)
    cBodyMaxPoint(2) = vBodyBounds(5)

    Set cLeftEdge = GetEdgeInViewForThisBody(False, False)
    Set cRightEdge = GetEdgeInViewForThisBody(False, True)
    Set cTopEdge = GetEdgeInViewForThisBody(True, True)
    Set cBottomEdge = GetEdgeInViewForThisBody(True, False)
        
    Set cStructuralFeat = GetStructuralMemberFeature
    
    If Not cStructuralFeat Is Nothing Then
    
        Dim StructuralMemberData As SldWorks.StructuralMemberFeatureData
        Set StructuralMemberData = cStructuralFeat.GetDefinition
        
        Set cSketchSegment = StructuralMemberData.GetPathSegmentAt(swBody)
        
        'cSketchSegment.Select4 False, Nothing
        
    End If
    
    Call UpdateSheetMaxMinDimensions(swView)
    
End Sub

Private Function GetStructuralMemberFeature() As SldWorks.Feature

    Dim vFeats As Variant
    vFeats = cBody.GetFeatures
    
    If Not IsEmpty(vFeats) Then
    
        Dim i As Integer
        For i = LBound(vFeats) To UBound(vFeats)
        
            Dim swFeat As SldWorks.Feature
            Set swFeat = vFeats(i)
            
            Debug.Print swFeat.Name
            Debug.Print swFeat.GetTypeName2
            
            If swFeat.GetTypeName2 = "WeldMemberFeat" Then
            
                Set GetStructuralMemberFeature = swFeat
                Exit For
                
            End If
        
        Next i

    End If

End Function

Private Sub UpdateSheetMaxMinDimensions(swView As SldWorks.View)

    Dim MinPoint As Variant
    MinPoint = GetComponentPointInSheetSpace(cComp, cBodyMinPoint, swView)

    Dim MaxPoint As Variant
    MaxPoint = GetComponentPointInSheetSpace(cComp, cBodyMaxPoint, swView)
    
    Call GetMaxMinPoint(MinPoint(0), MaxPoint(0), cxMin, cxMax)
    Call GetMaxMinPoint(MinPoint(1), MaxPoint(1), cyMin, cyMax)
    Call GetMaxMinPoint(MinPoint(2), MaxPoint(2), cZMin, cZMax)
    
    If Not cSketchSegment Is Nothing Then
    
        If cSketchSegment.GetType = swSketchSegments_e.swSketchLine Then
    
            Dim swSketchLine As SldWorks.SketchLine
            Set swSketchLine = cSketchSegment
            
            Dim vStartPoint As Variant
            vStartPoint = GetPointDataInSheetSpace(swSketchLine.GetStartPoint2, cSketchSegment.GetSketch, cComp, swView)
            
            Dim vEndPoint As Variant
            vEndPoint = GetPointDataInSheetSpace(swSketchLine.GetEndPoint2, cSketchSegment.GetSketch, cComp, swView)
            
            Call GetMaxMinPoint(vStartPoint(0), vEndPoint(0), cxMinSkPt, cxMaxSkPt)
            Call GetMaxMinPoint(vStartPoint(1), vEndPoint(1), cyMinSkPt, cyMaxSkPt)
            
            If Abs(vStartPoint(0) - vEndPoint(0)) <= 0.0001 Then
            
                cIsHorizontal = False
                cMid = vStartPoint(0)
                
        
            ElseIf Abs(vStartPoint(1) - vEndPoint(1)) <= 0.0001 Then
            
                cIsHorizontal = True
                cMid = vStartPoint(1)
                
            End If
            
        End If
    End If
    
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

Private Function GetEdgeInViewForThisBody(IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    Dim xFMin As Double
    Dim yFMin As Double
    Dim xFMax As Double
    Dim yFMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = cxMin
    vPointMin(1) = cyMin
    vPointMin(2) = cZMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = cxMax
    vPointMax(1) = cyMax
    vPointMax(2) = cZMax
    
    Call GetMaxMinPoint(vPointMin(0), vPointMax(0), xFMin, xFMax)
    Call GetMaxMinPoint(vPointMin(1), vPointMax(1), yFMin, yFMax)
    
    Dim Idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        Idx = 1
        If IsMax Then
        
            ValToMatch = yFMax
            
        Else
        
             ValToMatch = yFMin
             
        End If
        
    Else
    
        Idx = 0
        
        If IsMax Then
        
            ValToMatch = xFMax
            
        Else
        
             ValToMatch = xFMin
             
        End If
        
    End If

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = cView.GetVisibleEntities2(cComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = oBody.GetBody.GetEdges
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
            Dim swEdgeBody As SldWorks.Body2
            Set swEdgeBody = swEdge.GetBody
                
            If swEdgeBody.Name = cBody.Name Then
            
                'cView.SelectEntity swEdge, False

                Dim swCurve As SldWorks.Curve
                Set swCurve = swEdge.GetCurve
                
                If swCurve.IsLine Then
                
                    Dim vStartPoint As Variant
                    vStartPoint = swEdge.GetStartVertex.GetPoint
                    vStartPoint = GetComponentPointInSheetSpace(cComp, vStartPoint, cView)
                    
                    Dim vEndPoint As Variant
                    vEndPoint = swEdge.GetEndVertex.GetPoint
                    vEndPoint = GetComponentPointInSheetSpace(cComp, vEndPoint, cView)

                    If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.00001 And Abs(vStartPoint(Idx) - ValToMatch) <= 0.00001 Then
                        
                        Dim vCurveParam As Variant
                        vCurveParam = swEdge.GetCurveParams2
    
                        If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                            
                            TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                            Set GetEdgeInViewForThisBody = swEdge
                            
                        End If
                        
                    End If
                
                End If
                
            End If
            
        Next i

    End If

End Function






