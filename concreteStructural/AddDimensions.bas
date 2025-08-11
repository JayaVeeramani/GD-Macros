Attribute VB_Name = "AddDimensions"

Sub AddRebarDimensions(Dict As Scripting.Dictionary, LowerOrdDim As SldWorks.DisplayDimension, _
            HigherOrdDim As SldWorks.DisplayDimension, oConcreteComp As IComp, _
            MinParam As String, MaxParam As String, EdgeName As String, _
            swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, DimDict As Scripting.Dictionary)
    
    If Dict.Count > 0 Then
    
        Dim vItems As Variant
        vItems = Dict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim RebarList As IArrListObject
            Set RebarList = vItems(i)
            
            RebarList.SortItems MinParam, False
            
            Dim RebarItems As Variant
            RebarItems = RebarList.Items
            
            Dim oLowerRebar As IRebarBody
            Set oLowerRebar = RebarItems(0)
            
            Dim OrdRefDim As SldWorks.DisplayDimension
            Dim RefRebar As IRebarBody
            Dim SelXPoint As Double
            Dim SelYPoint As Double
            
            If UBound(RebarItems) = 0 Then
            
                If Abs(CallByName(oLowerRebar, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) > _
                     Abs(CallByName(oLowerRebar, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerRebar, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oLowerRebar, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set OrdRefDim = HigherOrdDim
                    Set RefRebar = oLowerRebar
                    SelXPoint = oLowerRebar.xMaxSketchPoint
                    SelYPoint = oLowerRebar.yMaxSketchPoint
                
                Else
                
                    Set OrdRefDim = LowerOrdDim
                    Set RefRebar = oLowerRebar
                    SelXPoint = oLowerRebar.xMinSketchPoint
                    SelYPoint = oLowerRebar.yMinSketchPoint
                    
                End If
            
            Else
                
                Dim oHigherRebar As IRebarBody
                Set oHigherRebar = RebarItems(UBound(RebarItems))

                If Abs(CallByName(oLowerRebar, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) > _
                     Abs(CallByName(oHigherRebar, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerRebar, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oHigherRebar, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set OrdRefDim = HigherOrdDim
                    Set RefRebar = oHigherRebar
                    SelXPoint = oLowerRebar.xMaxSketchPoint
                    SelYPoint = oLowerRebar.yMaxSketchPoint
                    
                Else
                
                    Set OrdRefDim = LowerOrdDim
                    Set RefRebar = oLowerRebar
                    SelXPoint = oLowerRebar.xMinSketchPoint
                    SelYPoint = oLowerRebar.yMinSketchPoint
                    
                End If
                
            End If
            
            Dim ValToCheck As Double
            If MinParam = "xMin" Then
                
                ValToCheck = RefRebar.yMinSketchPoint
                
            Else
            
                ValToCheck = RefRebar.xMinSketchPoint
            
            End If

            If False = CheckWhetherDimExists(OrdRefDim, DimDict, ValToCheck) Then
            
                Dim assyComponentName As String
                assyComponentName = swView.RootDrawingComponent.Component.Name2
        
                Dim assyDwgCompName As String
                assyDwgCompName = swView.RootDrawingComponent.Name
                
                Debug.Print RefRebar.SketchSegment.GetID(0)
                Debug.Print RefRebar.SketchSegment.GetID(1)

                Dim BoolStatus As Boolean
                BoolStatus = swDrawing.Extension.SelectByID2("Line" & RefRebar.SketchSegment.GetID(1) & "@" & _
                        RefRebar.SketchSegment.GetSketch.Name & "@" & assyDwgCompName & "@" & swView.Name & "/" & _
                        RefRebar.GetComponent.Name2 & "@" & assyDwgCompName, "EXTSKETCHSEGMENT", SelXPoint, SelYPoint, 0, False, 0, Nothing, 0)
                
                Dim oldDimCount As Integer
                oldDimCount = swView.GetDimensionCount4()
                
                Call AddToOrdinateDimension(OrdRefDim, RebarList.Count, swDrawing, swView)
                Call AddDimLocDataToDictionary(DimDict, OrdRefDim, RefRebar.xMinSketchPoint, _
                        RefRebar.yMinSketchPoint, Not (RefRebar.IsHorizontal), oldDimCount, swView)
                
            End If

        Next i
        
    End If
        
End Sub

Sub AddCompDimensions(Dict As Scripting.Dictionary, LowerOrdDim As SldWorks.DisplayDimension, _
            HigherOrdDim As SldWorks.DisplayDimension, oConcreteComp As IComp, _
            IsHorizontal As Boolean, MinParam As String, MaxParam As String, swDrawing As SldWorks.DrawingDoc, _
            swView As SldWorks.View, DimDict As Scripting.Dictionary, IsOrigin As Boolean)
    
    If Dict.Count > 0 Then
    
        Dim vItems As Variant
        vItems = Dict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim CompList As IArrListObject
            Set CompList = vItems(i)
            
            CompList.SortItems MinParam, False
            
            Dim CompItems As Variant
            CompItems = CompList.Items
            
            Dim oLowerComp As IComp
            Set oLowerComp = CompItems(0)
            
            Dim oRefComp As IComp
            Dim RefDim As SldWorks.DisplayDimension
            
            If UBound(CompItems) = 0 Then
            
                If Abs(CallByName(oLowerComp, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) < _
                     Abs(CallByName(oLowerComp, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerComp, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oLowerComp, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set oRefComp = oLowerComp
                    Set RefDim = LowerOrdDim
                
                Else
                
                    Set oRefComp = oLowerComp
                    Set RefDim = HigherOrdDim
                
                End If
            
            Else
                
                Dim oHigherComp As IComp
                Set oHigherComp = CompItems(UBound(CompItems))

                If Abs(CallByName(oLowerComp, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) < _
                     Abs(CallByName(oHigherComp, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerComp, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oHigherComp, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set oRefComp = oLowerComp
                    Set RefDim = LowerOrdDim
                
                Else
                
                    Set oRefComp = oHigherComp
                    Set RefDim = HigherOrdDim
                                        
                End If
                
            End If
            
                        
            Dim oldDimCount As Integer
            oldDimCount = swView.GetDimensionCount4()
            
            Dim xVal As Double
            Dim yVal As Double
            
            If IsOrigin Then
            
                Call SelectComponentOriginAndAddToOrdinateDimension(RefDim, oRefComp.GetComponent, CompList.Count, swDrawing, swView)
                xVal = oRefComp.xOrigin
                yVal = oRefComp.yOrigin
                
            Else
            
                Dim swEdge As SldWorks.Edge
                Set swEdge = GetEdgeInView(oRefComp, swView, IsHorizontal, False)
                
                swView.SelectEntity swEdge, False
                Call AddToOrdinateDimension(RefDim, CompList.Count, swDrawing, swView)
                
                xVal = oRefComp.xMin
                yVal = oRefComp.yMin
                
            End If

            Call AddDimLocDataToDictionary(DimDict, RefDim, xVal, yVal, Not (MinParam = "xMin"), oldDimCount, swView)

        Next i
        
    End If
        
End Sub

Function GetDimName(swDisplayDim As SldWorks.DisplayDimension) As String

    GetDimName = swDisplayDim.GetDimension2(0).FullName

End Function
Sub AddFoamDimensions(Dict As Scripting.Dictionary, LowerOrdDim As SldWorks.DisplayDimension, _
            HigherOrdDim As SldWorks.DisplayDimension, oConcreteComp As IComp, _
            MinParam As String, MaxParam As String, EdgeName As String, _
            swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, DimDict As Scripting.Dictionary)
    
    If Dict.Count > 0 Then
    
        Dim vItems As Variant
        vItems = Dict.Items
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
            
            Dim FoamList As IArrListObject
            Set FoamList = vItems(i)
            
            FoamList.SortItems MinParam, False
            
            Dim FoamItems As Variant
            FoamItems = FoamList.Items
            
            Dim oLowerFoam As IWeldBody
            Set oLowerFoam = FoamItems(0)
            
            Dim oRefFoam As IWeldBody
            Dim RefDim As SldWorks.DisplayDimension
            
            If UBound(FoamItems) = 0 Then
            
                If Abs(CallByName(oLowerFoam, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) < _
                     Abs(CallByName(oLowerFoam, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerFoam, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oLowerFoam, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set oRefFoam = oLowerFoam
                    Set RefDim = LowerOrdDim
                
                Else
                
                    Set oRefFoam = oLowerFoam
                    Set RefDim = HigherOrdDim
                
                End If
            
            Else
                
                Dim oHigherFoam As IWeldBody
                Set oHigherFoam = FoamItems(UBound(FoamItems))

                If Abs(CallByName(oLowerFoam, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) < _
                     Abs(CallByName(oHigherFoam, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet)) Or _
                      (Abs(CallByName(oLowerFoam, MinParam, VbGet) - CallByName(oConcreteComp, MinParam, VbGet)) - _
                     Abs(CallByName(oHigherFoam, MaxParam, VbGet) - CallByName(oConcreteComp, MaxParam, VbGet))) <= 0.0001 Then
                    
                    Set oRefFoam = oLowerFoam
                    Set RefDim = LowerOrdDim
                
                Else
                
                    Set oRefFoam = oHigherFoam
                    Set RefDim = HigherOrdDim
                                        
                End If
                
            End If
            
            Dim oldDimCount As Integer
            oldDimCount = swView.GetDimensionCount4()
            
            swView.SelectEntity CallByName(oRefFoam, EdgeName, VbGet), False
            Call AddToOrdinateDimension(RefDim, FoamList.Count, swDrawing, swView)
                
            Call AddDimLocDataToDictionary(DimDict, RefDim, CallByName(oRefFoam, "xMin", VbGet), _
                    CallByName(oRefFoam, "yMin", VbGet), Not (MinParam = "xMin"), oldDimCount, swView)

        Next i
        
    End If
        
End Sub

Sub AddDimLocDataToDictionary(ByRef DimDict As Scripting.Dictionary, swOrdinateDim As SldWorks.DisplayDimension, _
        xVal As Double, yVal As Double, IsXDim As Boolean, oldDimCount As Integer, swView As SldWorks.View)

    If oldDimCount + 1 = swView.GetDimensionCount4() Then
    
        Dim Val As Double
        If IsXDim Then
                    
            Val = xVal
    
        Else
                
            Val = yVal
                    
        End If
                
        Dim TempArrList As IArrList
        Dim KeyVal As String
        KeyVal = GetDimName(swOrdinateDim)
                
        If DimDict.Exists(KeyVal) Then
                
            Set TempArrList = DimDict.Item(KeyVal)
            TempArrList.AddtoList Val
                    
        Else
                
            Set TempArrList = New IArrList
            TempArrList.AddtoList Val
                    
            DimDict.Add KeyVal, TempArrList
                    
        End If
        
    End If


End Sub

Function CheckWhetherDimExists(swDim As SldWorks.DisplayDimension, DimDict As Scripting.Dictionary, ValToCheck As Double)
    
    CheckWhetherDimExists = False
    Dim KeyVal As String
    KeyVal = GetDimName(swDim)
    
    If DimDict.Exists(KeyVal) Then
    
        Dim ArrList As IArrList
        Set ArrList = DimDict.Item(KeyVal)
    
        Dim vArrItems As Variant
        vArrItems = ArrList.Items
        
        Dim Idx As Integer
        Dim PrevDiff As Double
        
        Dim i As Integer
        For i = LBound(vArrItems) To UBound(vArrItems)

            If Abs(vArrItems(i) - ValToCheck) <= 0.0001 Then
            
                CheckWhetherDimExists = True
                Exit For
                
            End If
            
            If i = 0 Then
            
                If ValToCheck - vArrItems(i) < 0 Then
                    
                    Exit For
                    
                End If
                
            Else
            
                If ValToCheck - vArrItems(i - 1) > 0 And ValToCheck - vArrItems(i) < 0 Then
                
                    Exit For
                    
                End If
                
            End If

        Next i
    
    End If

End Function
Private Sub AddQtyBracketsAndSuffixToDimension(swDisplayDim As SldWorks.DisplayDimension, Qty As Integer, Optional IsBrackets As Boolean, Optional suffixNote As String = "")

    If IsBrackets Then
    
         swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, "("
    
        If Qty > 1 Then
            
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X ("
                
        End If
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, ")" & vbCrLf & suffixNote
        
    Else
        
        If Qty > 1 Then
            
            swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextPrefix, Qty & "X "
                
        End If
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextSuffix, vbCrLf & suffixNote
        
    End If


End Sub

Function SelectAndAddOrdinateOrigin(swEnt As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
        xPos As Double, yPos As Double, Optional IsHorizontal As Boolean = False) As SldWorks.DisplayDimension

    swDrawing.ClearSelection2 True
    swDrawing.SetPickMode
    swView.SelectEntity swEnt, False
    
    If IsHorizontal Then
    
        swDrawing.InsertHorizontalOrdinate

    Else
    
         swDrawing.InsertVerticalOrdinate
         
    End If

    Call swDrawing.Extension.SelectByID2("", "VIEW", xPos, yPos, 0, False, 0, Nothing, 0)
    
    Dim swSelectMgr As SldWorks.SelectionMgr
    Set swSelectMgr = swDrawing.SelectionManager

    Set SelectAndAddOrdinateOrigin = swSelectMgr.GetSelectedObject6(1, -1)

End Function

Sub AddToOrdinateDimension(OrdDim As SldWorks.DisplayDimension, _
                Qty As Integer, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim PrevDimCount As Integer
    PrevDimCount = swView.GetDisplayDimensionCount
    
    Dim swDimAnn As SldWorks.Annotation
    Set swDimAnn = OrdDim.GetAnnotation
    swDimAnn.Select3 True, Nothing
    
    swDrawing.EditOrdinate
    
    If Qty > 1 Then
    
        If PrevDimCount + 1 = swView.GetDisplayDimensionCount Then
            
            Dim vDisplayDims As Variant
            vDisplayDims = swView.GetDisplayDimensions
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = GetLastAddDisplayDimension(swView)

            If Not swDisplayDim Is Nothing Then
        
                Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, Qty)
                
            End If
            
        End If
        
    End If

    swDrawing.SetPickMode
    swDrawing.ClearSelection2 True
    
End Sub

Sub SelectComponentOriginAndAddToOrdinateDimension(OrdDim As SldWorks.DisplayDimension, swComp As SldWorks.Component2, _
                Qty As Integer, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim PrevDimCount As Integer
    PrevDimCount = swView.GetDisplayDimensionCount
    
    Dim swDimAnn As SldWorks.Annotation
    Set swDimAnn = OrdDim.GetAnnotation
    swDimAnn.Select3 False, Nothing

    Call SelectComponentOrigin(swComp, swDrawing, swView, True)
    swDrawing.EditOrdinate
    
    If Qty > 1 Then
    
        If PrevDimCount + 1 = swView.GetDisplayDimensionCount Then
            
            Dim vDisplayDims As Variant
            vDisplayDims = swView.GetDisplayDimensions
            
            Dim swDisplayDim As SldWorks.DisplayDimension
            Set swDisplayDim = GetLastAddDisplayDimension(swView)

            If Not swDisplayDim Is Nothing Then
        
                Call AddQtyBracketsAndSuffixToDimension(swDisplayDim, Qty)
                
            End If
            
        End If
        
    End If

    swDrawing.SetPickMode
    swDrawing.ClearSelection2 True
    
End Sub

Function SelectComponentOrigin(swComp As SldWorks.Component2, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, Append As Boolean) As Boolean
    
    Dim assyComponentName As String
    assyComponentName = swView.RootDrawingComponent.Component.Name2
    
    Dim assyDwgCompName As String
    assyDwgCompName = swView.RootDrawingComponent.Name

    
    Debug.Print "Point1@Origin@" & assyDwgCompName & "@" & swView.Name & "/" & swComp.Name2 & "@" & assyComponentName
    SelectComponentOrigin = swDrawing.Extension.SelectByID2("Point1@Origin@" & assyDwgCompName & "@" & swView.Name _
        & "/" & swComp.Name2 & "@" & assyComponentName, "EXTSKETCHPOINT", 0, 0, 0, Append, 0, Nothing, 0)

End Function


Function GetLastAddDisplayDimension(swView As SldWorks.View) As SldWorks.DisplayDimension

    Dim vDisplayDims As Variant
    vDisplayDims = swView.GetDisplayDimensions
    
    Dim DimNameToFind As String
    DimNameToFind = "D" & swView.GetDisplayDimensionCount + 4
    
    Dim i As Integer
    For i = UBound(vDisplayDims) To LBound(vDisplayDims) Step -1
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = vDisplayDims(i)
        
        Dim swDimAnn As SldWorks.Annotation
        Set swDimAnn = swDisplayDim.GetAnnotation
        
        If swDimAnn.GetName = DimNameToFind Then
        
            Set GetLastAddDisplayDimension = swDisplayDim
            Exit For
            
        End If
    
    Next i

End Function
Function SelectAndAddDimension(swEnt1 As SldWorks.Entity, swEnt2 As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View, Optional IsDual As Boolean = True) As SldWorks.DisplayDimension

    If Not (swEnt1 Is Nothing) And Not (swEnt2 Is Nothing) Then

        swDrawing.ClearSelection2 True
        
        swView.SelectEntity swEnt1, False
        swView.SelectEntity swEnt2, True

        Set SelectAndAddDimension = swDrawing.AddDimension2(xPos, yPos, 0)

        If Not SelectAndAddDimension Is Nothing Then

            SelectAndAddDimension.CenterText = True

            If IsDual Then

                SelectAndAddDimension.SetDual2 False, False

            End If

        End If

    End If

End Function

Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
        
        swView.SelectEntity swEdge, False
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub
