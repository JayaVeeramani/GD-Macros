Attribute VB_Name = "CrossMark"

Function sdfsdf(swSketch As SldWorks.Sketch) As IArrListObject

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
            
                Dim swContourSketch As IContourSketch
                Set swContourSketch = New IContourSketch
                
                swContourSketch.Initialize swContour
            
                If swContourSketch.isRectangular Then
                
                    vContourArrList.AddtoList swContourSketch
                    
                End If

            End If
        
        Next
        
    End If
    
    Set GetSketchContours = vContourArrList

End Function

Sub AddCrossMarkForContours(vContours As Variant, swDrawing As SldWorks.DrawingDoc, _
            swFeat As SldWorks.Feature, swSketch As SldWorks.Sketch, swView As SldWorks.View, AssyName As String)
    
    swView.FocusLocked = True
    
    Dim OffsetX As Double
    Dim OffsetY As Double

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

    Dim i As Integer
    For i = LBound(vContours) To UBound(vContours)
    
        Dim swSketchContour As IContourSketch
        Set swSketchContour = vContours(i)
        
        Dim swSketchManager As SldWorks.SketchManager
        Set swSketchManager = swDrawing.SketchManager
        
        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.bottomLeftPoint, swSketchContour.topRightPoint, SelectionString2, SelectionString3)
        Call AddSketchSegmentsAndConstraints(swDrawing, swSketchManager, OffsetX, OffsetY, swSketchContour.TopLeftPoint, swSketchContour.BottomRightPoint, SelectionString2, SelectionString3)
        
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

Sub AddCrossMarkForDoor(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
                swBottomEdge As SldWorks.Edge, DoorList As IArrListObject)

    
    Dim vDoorItems As Variant
    vDoorItems = DoorList.Items
    
    
    Dim skSegment As SketchSegment
    Set skSegment = swSketchManager.CreateLine(FirstPoint.X - OffsetX, FirstPoint.Y - OffsetY, FirstPoint.Z, _
                        SecondPoint.X - OffsetX, SecondPoint.Y - OffsetY, SecondPoint.Z)
    skSegment.ConstructionGeometry = True
    skSegment.Layer = Layername
    
    Dim skLine As SldWorks.SketchLine
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
