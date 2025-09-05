Attribute VB_Name = "EdgeSelection"

Function GetEdgeInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
  
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = oComp.xMin
    vPointMin(1) = oComp.yMin
    vPointMin(2) = oComp.zMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = oComp.xMax
    vPointMax(1) = oComp.yMax
    vPointMax(2) = oComp.zMax
    
    Call GetMaxMinPoint(vPointMin(0), vPointMax(0), xMin, xMax)
    Call GetMaxMinPoint(vPointMin(1), vPointMax(1), yMin, yMax)
    
    'Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
    Dim Idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        Idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        Idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If
    
    Dim swComp As SldWorks.Component2
    Set swComp = oComp.GetComponent
    

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
    Else
    
        vEnts = GetComponentEdges(swComp)
        
    End If

    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
            Dim IsSelected As Boolean
            '
            
            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsLine Then
            
                Dim vStartPoint As Variant
                vStartPoint = swEdge.GetStartVertex.GetPoint
                vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)
  
                
                If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.0015875 * swView.ScaleDecimal And Abs(vStartPoint(Idx) - ValToMatch) <= 0.0015875 * swView.ScaleDecimal And _
                        Abs(vStartPoint(2) - vEndPoint(2)) <= 0.0015875 * swView.ScaleDecimal Then
                    
                    Dim vCurveParam As Variant
                    vCurveParam = swEdge.GetCurveParams2
                    
                    If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                        
                        TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                        Set GetEdgeInView = swEdge
                        
                    End If
                    
                End If
            
            End If
            
        Next i

    End If

End Function

Function GetPointInView(oComp As IComp, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean) As SldWorks.Vertex

    Dim SortParam As String
    If IsHorizontal Then
        
        SortParam = "xVal"

    Else
    
        SortParam = "yVal"

    End If
    
    Dim VertexPointList As IArrListObject
    Set VertexPointList = GetVertexPointList(oComp, swView)
    
    If VertexPointList.Count > 0 Then
        
        VertexPointList.SortItems SortParam, IsMax
        Set GetPointInView = VertexPointList.Items(0).Vertex
    
    End If

End Function

Function GetVertexPointList(oComp As IComp, swView As SldWorks.View) As IArrListObject
    
    Set GetVertexPointList = New IArrListObject
    
    Dim swComp As SldWorks.Component2
    Set swComp = oComp.GetComponent

    Dim vEnts As Variant
    vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Vertex)
    
   If Not IsEmpty(vEnts) Then
        
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swVertex As SldWorks.Vertex
            Set swVertex = vEnts(i)
            
            Dim oVertex As IVertexPt
            Set oVertex = New IVertexPt
            
            oVertex.Initialize swVertex, swComp, swView
            
            GetVertexPointList.AddtoList oVertex

        Next i

    End If
    
End Function

Function GetExtremePointInView(oComp As IComp, swView As SldWorks.View, _
    IsTop As Boolean, IsRight As Boolean) As SldWorks.Vertex
    
    Dim VertexPointList As IArrListObject
    Set VertexPointList = GetVertexPointList(oComp, swView)
    
    If VertexPointList.Count > 0 Then
        
        VertexPointList.SortItems "xVal", IsRight
        VertexPointList.SortItems "yVal", IsTop
        
        Set GetExtremePointInView = VertexPointList.Items(0).Vertex
        
    
    End If

End Function


Function GetComponentEdges(swComp As SldWorks.Component2)
    
    Dim TempEdges As Variant
    
    Dim vBodies As Variant
    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, swBodyInfo_e.swNormalBody_e)
    
    Dim i As Integer
    Dim j As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swBody As SldWorks.Body2
        Set swBody = vBodies(i)
        
        Dim vEdges As Variant
        vEdges = swBody.GetEdges
        
        If i = 0 Then
            
            TempEdges = vEdges
            
        Else
            
            TempEdges = CombineArr(TempEdges, vEdges)
            
        End If
    
    Next i
    
    GetComponentEdges = TempEdges

End Function

Function CombineArr(ByVal MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
    CombineArr = MainArr
    
End Function


Function GetEdgeInViewForBody(swComp As SldWorks.Component2, oBody As Object, swView As SldWorks.View, _
    IsHorizontal As Boolean, IsMax As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge

    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
    
    Dim vPointMin(2) As Double
    vPointMin(0) = oBody.xMin
    vPointMin(1) = oBody.yMin
    vPointMin(2) = oBody.zMin
    
    Dim vPointMax(2) As Double
    vPointMax(0) = oBody.xMax
    vPointMax(1) = oBody.yMax
    vPointMax(2) = oBody.zMax
    
    Call GetMaxMinPoint(vPointMin(0), vPointMax(0), xMin, xMax)
    Call GetMaxMinPoint(vPointMin(1), vPointMax(1), yMin, yMax)
    
    Dim Idx As Integer
    Dim ValToMatch As Double
    If IsHorizontal Then
        
        Idx = 1
        If IsMax Then
        
            ValToMatch = yMax
            
        Else
        
             ValToMatch = yMin
             
        End If
        
    Else
    
        Idx = 0
        
        If IsMax Then
        
            ValToMatch = xMax
            
        Else
        
             ValToMatch = xMin
             
        End If
        
    End If

     Dim TempLength As Double
     TempLength = 0
        

    Dim vEnts As Variant
    If CheckAllVisibleEdgesOnly Then
    
        vEnts = swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Edge)
        
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
                
            If swEdgeBody.Name = oBody.GetBody.Name Then

                Dim swCurve As SldWorks.Curve
                Set swCurve = swEdge.GetCurve
                
                Dim Bool As Boolean
                Bool = swView.SelectEntity(swEdge, False)
                
                If swCurve.IsLine Then
                
                    Dim vStartPoint As Variant
                    vStartPoint = swEdge.GetStartVertex.GetPoint
                    vStartPoint = GetComponentPointInSheetSpace(swComp, vStartPoint, swView)
                    
                    Dim vEndPoint As Variant
                    vEndPoint = swEdge.GetEndVertex.GetPoint
                    vEndPoint = GetComponentPointInSheetSpace(swComp, vEndPoint, swView)

                    If Abs(vStartPoint(Idx) - vEndPoint(Idx)) <= 0.00001 And Abs(vStartPoint(Idx) - ValToMatch) <= 0.00001 Then
                        
                        Dim vCurveParam As Variant
                        vCurveParam = swEdge.GetCurveParams2
    
                        If swCurve.GetLength2(vCurveParam(6), vCurveParam(7)) > TempLength Then
                            
                            TempLength = swCurve.GetLength2(vCurveParam(6), vCurveParam(7))
                            Set GetEdgeInViewForBody = swEdge
                            
                        End If
                        
                    End If
                
                End If
                
            End If
            
        Next i

    End If

End Function

 Function SelectEntityWithSelectData(swEnt As Object, swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                SelXPos As Double, SelYPos As Double) As Boolean
                
    If Not swEnt Is Nothing Then

        Dim swSelectData As SldWorks.SelectData
        Set swSelectData = CreateSelectData(swView, swDrawing, SelXPos, SelYPos)
    
        Dim swEntity As SldWorks.Entity
        Set swEntity = swEnt
    
        SelectEntityWithSelectData = swEntity.Select4(False, swSelectData)
        
    End If
    
End Function

Function CreateSelectData(swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc, _
                SelXPos As Double, SelYPos As Double) As SldWorks.SelectData
 
    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager
    
    Set CreateSelectData = swSelectMgr.CreateSelectData

    CreateSelectData.View = swView
    CreateSelectData.X = SelXPos
    CreateSelectData.Y = SelYPos

End Function
