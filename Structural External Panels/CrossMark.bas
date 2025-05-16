Attribute VB_Name = "CrossMark"

Function GetSketchContours(swSketch As SldWorks.Sketch, swComp As SldWorks.Component2, swView As SldWorks.View) As Variant

    Dim IsInit As Boolean
    IsInit = True
    
    Dim vContours() As IContourSketch
    
    If swSketch.GetSketchContourCount > 0 Then
        
        Dim vSketchContours As Variant
        vSketchContours = swSketch.GetSketchContours
        
        Dim vContourArrList As IArrListObject
        Set vContourArrList = New IArrListObject
        
        Dim i As Integer
        For i = LBound(vSketchContours) To UBound(vSketchContours)
                
            Dim swContour As SldWorks.SketchContour
            Set swContour = vSketchContours(i)
            
            If swContour.IsClosed And swContour.GetSketchSegmentsCount = 4 Then
                
                Dim vSketchSegs As Variant
                vSketchSegs = swContour.GetSketchSegments
                
                Dim swSketchSegment As SldWorks.SketchSegment
                Set swSketchSegment = vSketchSegs(0)
                
                If Not swSketchSegment.ConstructionGeometry Then
                
                    Dim swContourSketch As IContourSketch
                    Set swContourSketch = New IContourSketch
                    
                    
                    
                    swContourSketch.Initialize swContour, swSketch, swComp, swView
                
                    If swContourSketch.isRectangular Then
                    
                        vContourArrList.AddtoList swContourSketch
                        
                    End If
                    
                End If

            End If
        
        Next
        
    End If
    
    GetSketchContours = vContourArrList.Items

End Function

Sub AddCrossMarkAndDimensionsForContours(vContours As Variant, swDrawing As SldWorks.DrawingDoc, _
            swFeat As SldWorks.Feature, swSketch As SldWorks.Sketch, swView As SldWorks.View, _
                AssyName As String, oSubAssy As ISubAssy)
    
    swView.FocusLocked = True
    
    Dim OffsetX As Double
    Dim OffsetY As Double
    
    Dim ContourList As IArrListObject
    Set ContourList = New IArrListObject
    
    ContourList.SortItems "xMin", False

    Call GetOffsetValues(OffsetX, OffsetY, swDrawing, swView)
    
    Dim swEntity As SldWorks.Entity
    Set swEntity = swSketch
    
    Dim assyComponentName As String
    assyComponentName = swView.RootDrawingComponent.Component.Name2
    
    Dim assyDwgcompName As String
    assyDwgcompName = swView.RootDrawingComponent.Name
    
    Dim viewName As String
    viewName = swView.Name
    
    Dim sketchName As String
    sketchName = swFeat.Name
    
    Dim SelectionString2 As String
    SelectionString2 = sketchName & "@" & assyDwgcompName & "@" & viewName
    
    Dim SelectionString3 As String
    SelectionString3 = ExtractCompNameForSelectByID(assyComponentName, AssyName)
    
    Dim swSketchToModelTransform As SldWorks.MathTransform
    Set swSketchToModelTransform = swSketch.ModelToSketchTransform.Inverse
    
    Debug.Print AssyName
    
            
    Dim swSketchManager As SldWorks.SketchManager
    Set swSketchManager = swDrawing.SketchManager
    
    Dim ConsolidatedHorizontalDict As Scripting.Dictionary
    Set ConsolidatedHorizontalDict = New Scripting.Dictionary
    


    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
    
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)

        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.bottomLeftPoint, swSketchContour.topRightPoint, SelectionString2, SelectionString3)
        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.TopLeftPoint, swSketchContour.BottomRightPoint, SelectionString2, SelectionString3)
        
        Dim xMin As Double
        xMin = Round(swSketchContour.xMin, 4)
        

        
        Call CheckAddToDict(ConsolidatedHorizontalDict, xMin, swSketchContour)
        
        
'        Dim swDisplayDim As SldWorks.DisplayDimension
'
'        If Abs(oSubAssy.StartComp.xMin - swSketchContour.xMin) < Abs(oSubAssy.EndComp.xMax - swSketchContour.xMax) Then
'
'            Call SelectLine(swDrawing, swSketchContour.LeftSketchLine, SelectionString2, SelectionString3, False)
'            swView.SelectEntity oSubAssy.StartEdge, True
'
'            Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.StartComp.xMin + 0.001, swSketchContour.yMin - 0.005, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'
'            Call SelectLine(swDrawing, swSketchContour.RightSketchLine, SelectionString2, SelectionString3, False)
'            Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMax + 0.005, swSketchContour.yMin + 0.001, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'        Else
'
'            Call SelectLine(swDrawing, swSketchContour.RightSketchLine, SelectionString2, SelectionString3, False)
'            swView.SelectEntity oSubAssy.EndEdge, True
'
'            Set swDisplayDim = swDrawing.AddHorizontalDimension2(oSubAssy.EndComp.xMax - 0.001, swSketchContour.yMin - 0.005, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'            Call SelectLine(swDrawing, swSketchContour.LeftSketchLine, SelectionString2, SelectionString3, False)
'            Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMin - 0.005, swSketchContour.yMin + 0.001, 0)
'            Call CenterAndManualParanthesis(swDisplayDim)
'
'        End If
'
'        Call SelectLine(swDrawing, swSketchContour.BottomSketchLine, SelectionString2, SelectionString3, False)
'        swView.SelectEntity oSubAssy.BottomEdge, True
'        Set swDisplayDim = swDrawing.AddVerticalDimension2(swSketchContour.xMax + 0.005, oSubAssy.StartComp.yMin + 0.001, 0)
'        Call CenterAndManualParanthesis(swDisplayDim)
'
'        Call SelectLine(swDrawing, swSketchContour.TopSketchLine, SelectionString2, SelectionString3, False)
'        Set swDisplayDim = swDrawing.AddHorizontalDimension2(swSketchContour.xMin + 0.001, swSketchContour.yMax + 0.005, 0)
'        Call CenterAndManualParanthesis(swDisplayDim)
        
    Next i
    
    ContourList.SortItems "yMin", False
    
    Dim ConsolidatedVerticalDict As Scripting.Dictionary
    Set ConsolidatedVerticalDict = ConsolidateContoursVertically(ContourList.Items)
    
    Call AddVerticalDimension(ConsolidatedVerticalDict, oSubAssy, swDrawing, swView, SelectionString2, SelectionString3)
 
End Sub

Function ConsolidateContoursVertically(vContours As Variant) As Scripting.Dictionary

    ConsolidateContoursVertically = New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
    
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)
    
        Dim yMin As Double
        yMin = Round(swSketchContour.yMin, 4)
        
        Call CheckAddToDict(ConsolidateContoursVertically, yMin, swSketchContour)
        
    Next i

End Function

Sub AddVerticalDimension(Dict As Scripting.Dictionary, oSubAssy As ISubAssy, _
        swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, SelectionString2 As String, SelectionString3 As String)

    Dim vKeys As Variant
    vKeys = Dict.Keys
    
    Dim i As Integer
    For i = LBound(vKeys) To UBound(vKeys)
    
        Dim ArrList As IArrListObject
        Set ArrList = Dict.Item(vKeys(i))
        
        ArrList.SortItems "xMin", True
        
        Dim vContours As Variant
        vContours = ArrList.Items
        
        Dim widthContourDict As Scripting.Dictionary
        
        Dim widthContourQtyDict As Scripting.Dictionary
        Set widthContourQtyDict = New Scripting.Dictionary
        
        Set widthContourDict = ConsolidateBasedOnWidth(vContours, widthContourQtyDict)

        
        Dim vWidthKeys As Variant
        vWidthKeys = widthContourDict.Keys
        
        Dim j As Integer
        For j = LBound(vWidthKeys) To UBound(vWidthKeys)
        
            Dim swSketchContour As IContourSketch
            Set swSketchContour = widthContourDict.Item(vWidthKeys(i))
            
            
            
            If widthContourQtyDict.Item(vWidthKeys(i)) > 1 Then
            
            End If
            
        
        
        Next j

    Next i
    
    
End Sub

Function ConsolidateBasedOnWidth(vContours As Variant, ByRef ConsolidatedQtyDict As Scripting.Dictionary) As Scripting.Dictionary
    
    Set ConsolidateBasedOnWidth = New Scripting.Dictionary
    
    Dim j As Integer
    For j = LBound(vContours) To UBound(vContours)
        
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(j)
        
        Dim keyVal As Double
        keyVal = Round(swSketchContour.Width, 4)
        
        If ConsolidateBasedOnWidth.Exists(keyVal) Then
        
            ConsolidatedQtyDict.Item(keyVal) = ConsolidatedQtyDict.Item(keyVal) + 1
            
        Else
        
            ConsolidateBasedOnWidth.Add , swSketchContour
            ConsolidatedQtyDict.Add keyVal, 1
        
        End If

    Next j
 
End Function

Sub CheckAddToDict(ByRef Dict As Scripting.Dictionary, keyVal As Double, swSketchContour As IContourSketch)
    
    Dim ArrList As IArrListObject
    
    If Dict.Exists(keyVal) Then

        Set ArrList = Dict.Item(keyVal)
        ArrList.AddtoList swSketchContour
        
    Else
    
        If Dict.Count > 0 Then
            
            If Abs(Dict.Keys(UBound(Dict.Keys)) - keyVal) <= 0.001 Then
            
                Set ArrList = Dict.Item(keyVal)
                ArrList.AddtoList swSketchContour
                
            Else
                
                Set ArrList = New IArrListObject
                ArrList.AddtoList swSketchContour
                
                Dict.Add keyVal, ArrList
                
            End If
            
        Else
        
            Set ArrList = New IArrListObject
            ArrList.AddtoList swSketchContour
            
            Dict.Add keyVal, ArrList
            
        End If
        
    
    End If
    
End Sub

Sub HorizontalDimensionsForContours(vContours As Variant, swDrawing As SldWorks.DrawingDoc, swSketchMgr As SldWorks.SketchManager)

    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
        
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)
        
        If i = 0 Then
            
            
        Else
        
        
        End If
    
    Next i

End Sub

Sub GetOffsetValues(ByRef OffsetX As Double, ByRef OffsetZ As Double, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim values As Variant
    values = swView.Position

    Dim dOrigPt(2) As Double
    dOrigPt(0) = 0: dOrigPt(1) = 0: dOrigPt(2) = 0

    Dim vMathPt  As Variant
    vMathPt = GetTransformPoint(dOrigPt, swView.ModelToViewTransform)

    Dim swScale As Double
    swScale = swView.ScaleDecimal

    OffsetX = (values(0) - vMathPt(0)) / (swScale) ' * 0.0254)
    OffsetZ = (values(1) - vMathPt(1)) / (swScale) ' * 0.0254)

End Sub

Sub AddSketchSegmentsAndConstraints(swDrawing As SldWorks.DrawingDoc, swSketchManager As SldWorks.SketchManager, OffsetX As Double, OffsetY As Double, _
                FirstPoint As SldWorks.sketchPoint, SecondPoint As SldWorks.sketchPoint, SelectionString2 As String, SelectionString3 As String)

    Dim x1Val As Double
    Dim x2Val As Double
    Dim y1Val As Double
    Dim y2Val As Double
    
    Dim skSegment As SketchSegment
    Set skSegment = swSketchManager.CreateLine(FirstPoint.X - OffsetX, FirstPoint.Y - OffsetY, FirstPoint.Z, _
                        SecondPoint.X - OffsetX, SecondPoint.Y - OffsetY, SecondPoint.Z)
    skSegment.ConstructionGeometry = True
    skSegment.Layer = Layername
    
    Dim skLine As SldWorks.sketchLine
    Set skLine = skSegment

    Call AddConstraint(swDrawing, FirstPoint, skLine.GetStartPoint2, FirstPoint.X, FirstPoint.Y, FirstPoint.Z, SelectionString2, SelectionString3)
    Call AddConstraint(swDrawing, SecondPoint, skLine.GetEndPoint2, SecondPoint.X, SecondPoint.Y, SecondPoint.Z, SelectionString2, SelectionString3)
    
End Sub

Sub AddConstraint(swDrawing As SldWorks.DrawingDoc, sketchPoint As SldWorks.sketchPoint, linePoint As SldWorks.sketchPoint, xVal, yVal, _
    zVal, SelectionString2 As String, SelectionString3 As String)

    Dim bool As Boolean
    bool = swDrawing.Extension.SelectByID2("Point" & sketchPoint.GetID(1) & "@" & SelectionString2 _
        & SelectionString3, "EXTSKETCHPOINT", xVal, yVal, zVal, False, 0, Nothing, 0)

    If bool Then
        
        linePoint.Select4 True, Nothing
        swDrawing.SketchAddConstraints "sgCOINCIDENT"

    End If
    
 
End Sub

Function SelectLine(swDrawing As SldWorks.DrawingDoc, sketchLine As SldWorks.SketchSegment, _
        SelectionString2 As String, SelectionString3 As String, Append As Boolean) As Boolean

    SelectLine = swDrawing.Extension.SelectByID2("Line" & sketchLine.GetID(1) & "@" & SelectionString2 _
        & SelectionString3, "EXTSKETCHSEGMENT", 0, 0, 0, Append, 0, Nothing, 0)

End Function


Function ExtractCompNameForSelectByID(TopLevelCompName As String, ChildName As String)

    Dim vChildNames As Variant
    vChildNames = Split(ChildName, "/")
    
    Dim TempString As String
    
    Dim i As Integer
    For i = LBound(vChildNames) To UBound(vChildNames)
    
        If i = LBound(vChildNames) Then
    
            TempString = "/" & vChildNames(i) & "@" & TopLevelCompName
            
        Else
        
            TempString = TempString & "/" & vChildNames(i) & "@" & Left(vChildNames(i - 1), InStrRev(vChildNames(i - 1), "-") - 1)
            
        End If

    
    Next i
    
    Debug.Print TempString
    
    ExtractCompNameForSelectByID = TempString
    
End Function

Sub CenterAndManualParanthesis(swDisplayDim As SldWorks.DisplayDimension, Optional BottomText As String = "")

    swDisplayDim.CenterText = True
    
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
    swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")"

    If Not (BottomText = "") Then
    
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, BottomText
    
    End If
    

End Sub
