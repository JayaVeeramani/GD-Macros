VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   1440
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   3408
   OleObjectBlob   =   "HideShowForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "HideShowForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False


Option Explicit

Dim swMathUtility As SldWorks.MathUtility
Dim swSketchMgr As SldWorks.SketchManager
Dim xDirectionVector(2) As Double
Dim yDirectionVector(2) As Double
Dim zDirectionVector(2) As Double


Private Sub CloseButton_Click()
    
    Unload Me
    
End Sub

Private Function GetOppositeVector(Dir As Variant) As Double()

    Dim Temp(2) As Double
    Dim i As Integer
    For i = LBound(Dir) To UBound(Dir)
    
        Temp(i) = -1 * Dir(i)
    
    Next i
    
    GetOppositeVector = Temp
End Function

Function SelectEntity(swEnt As Object, Append As Boolean, swView As SldWorks.View) As Boolean 'swView As SldWorks.View

    Dim swEntity As SldWorks.Entity
    Set swEntity = swEnt
    
    'SelectEntity = swEntity.Select4(Append, Nothing)
    
    SelectEntity = swView.SelectEntity(swEnt, Append)
    
End Function

Private Sub CreateButton_Click()

    xDirectionVector(0) = 1
    xDirectionVector(1) = 0
    xDirectionVector(2) = 0
    
    yDirectionVector(0) = 0
    yDirectionVector(1) = 1
    yDirectionVector(2) = 0
    
    zDirectionVector(0) = 0
    zDirectionVector(1) = 0
    zDirectionVector(2) = 1

    Me.Hide
    
    Dim wallName As String
    wallName = WallDrawingForm.WallNameComboBox.Value
    
    Dim ProjectNo As String
    ProjectNo = WallDrawingForm.ProjectNoBox.Value
    
    Unload WallDrawingForm
    
    Dim viewName As String
    viewName = GetViewName(wallName)
    
    Set swMathUtility = swApp.GetMathUtility

    Dim swViewNormalVector As SldWorks.MathVector
    Set swViewNormalVector = swMathUtility.CreateVector(GetViewVector(viewName))
    
    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\ASSEMBLY DRAWING.drwdot", 0, 0, 0)
    
    Set swSketchMgr = swDrawing.SketchManager
    
    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)
    Call AddLegendBlocks(swDrawing, swSheet)
    
    Dim swFrontView As SldWorks.View
    Set swFrontView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), viewName, 0.21593179, 0.19172741, 0)
    
    Dim IsZChannelExists As Boolean
    Dim ViewWidth As Double
    Dim ViewHeight As Double
    Dim MaxHeightComp As IComp
    Dim CompList As IArrListObject
    Set CompList = GetComponentsSortedWithYPosition(swFrontView, swDrawing, swViewNormalVector, ViewWidth, ViewHeight, MaxHeightComp, IsZChannelExists)
    
    Dim IsMultipleAssembly As Boolean
    IsMultipleAssembly = CheckForMultipleAssembly(ViewWidth / swFrontView.ScaleDecimal, ViewHeight / swFrontView.ScaleDecimal)
    
    Dim subAssyEndComponents As Variant
    If IsMultipleAssembly Then

        Call ActivateDrawingDocument(swTopLevelModel)
        SubAssyForm.Show vbModeless
        
        IsSubAssyFormClicked = False
        Do While IsSubAssyFormClicked = False
            
            DoEvents
            
        Loop
        
        subAssyEndComponents = GetSelectedComponents
        Call ActivateDrawingDocument(swDrawing)
        
    End If
    
    Dim swBottomView As SldWorks.View
    Set swBottomView = ScaleAndInsertBottomView(swDrawing, swFrontView, ViewWidth, ViewHeight)
    
    Call CleanUpActivateAndAddViewLabel(swDrawing, swFrontView, wallName)
    
    Dim FlatCompList As Variant
    Dim DetailedCompList As Variant
    Dim MaxCompHeight As Double
    DetailedCompList = GetComponentsSortedWithXPosition(CompList.Items, FlatCompList, swFrontView, MaxCompHeight)

    Dim vConsolidatedList As Variant
    
    Dim HVACList As IArrListObject
    Set HVACList = New IArrListObject
    
    Dim DoorList As IArrListObject
    Set DoorList = New IArrListObject
    
    vConsolidatedList = GetConsolidatedList(DetailedCompList, DoorList, HVACList)
    
    swDrawing.ActivateView swFrontView.Name
    
    Dim IsMakeUpExists As Boolean
    'Call AddCallouts(vConsolidatedList, swDrawing, swFrontView, MaxCompHeight, IsMakeUpExists)

    Dim Is12GAPanelExists As Boolean
    'Is12GAPanelExists = Add12GACircles(FlatCompList, swDrawing, swBottomView)

    Call UpdateBottomViewPosition(FlatCompList, swDrawing, swBottomView)
    Call AddStructuralNotes(swDrawing, swSheet, Is12GAPanelExists)
    
    Dim swLeftEdge As SldWorks.Edge
    Dim swRightEdge As SldWorks.Edge
    Call AddDimensionInFrontView(swFrontView, FlatCompList, DetailedCompList, swDrawing, MaxHeightComp, swLeftEdge, swRightEdge)
    
    Dim FlatCompDict As Scripting.Dictionary
    Dim CompNoDict As New Scripting.Dictionary
    Set FlatCompDict = GetCompDictionary(FlatCompList, CompNoDict)
    
    Dim SubAssyList As IArrListObject
    If Not IsEmpty(subAssyEndComponents) Then
        
        Dim vSubAssyComponentsIdx As Variant
        vSubAssyComponentsIdx = GetSubAssyComponentsIndexSorted(subAssyEndComponents, CompNoDict)
   
        Set SubAssyList = AddSplitLines(vSubAssyComponentsIdx, swDrawing, swFrontView, FlatCompDict, CompNoDict, True, swLeftEdge, swRightEdge, False)
        Call AddDimensionNames(SubAssyList, wallName)
        
        Call AddSplitLines(vSubAssyComponentsIdx, swDrawing, swBottomView, FlatCompDict, CompNoDict, False, swLeftEdge, swRightEdge)
   
    End If
    
    Unload Me

End Sub

Private Sub AddDimensionNames(SubAssyList As IArrListObject, wallName As String)
    
    Dim CloneList As IArrListObject
    Set CloneList = New IArrListObject
    
    Set CloneList = SubAssyList.Clone

    If InStr(wallName, "Wall") > 0 Then
        
        CloneList.SortItems "AssyLength"
    
    End If
    
    Dim i As Integer
    Dim vSubAssy As Variant
    vSubAssy = CloneList.Items
    
    For i = LBound(vSubAssy) To UBound(vSubAssy)
    
        Dim oSubAssy As ISubAssy
        Set oSubAssy = vSubAssy(i)
        
        Dim swDisplayDim As SldWorks.DisplayDimension
        Set swDisplayDim = oSubAssy.Dimension
        
        swDisplayDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, UCase(wallName) & i + 1 & vbCrLf & "(XX.XX sq.ft)"
    
    Next i
    
End Sub

Private Function GetSubAssyComponentsIndexSorted(vComps As Variant, CompNoDict As Scripting.Dictionary)
    
    Dim vIdxArr As IArrList
    Set vIdxArr = New IArrList
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        vIdxArr.AddtoList CompNoDict(vComps(i).Name2)
        
    Next i
    
    vIdxArr.SortItems False
    GetSubAssyComponentsIndexSorted = vIdxArr.Items

End Function

Private Function AddSplitLines(vCompsIdx As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
        CompDict As Scripting.Dictionary, CompNoDict As Scripting.Dictionary, IsFrontView As Boolean, _
        swLeftEdge As SldWorks.Edge, swRightEdge As SldWorks.Edge, Optional VisibleEdgesOnly As Boolean = True) As IArrListObject

    swDrawing.ActivateView swView.Name
    
    Dim vOutline As Variant
    vOutline = swView.GetOutline
    
    Dim i As Integer
    Dim NextAssyStartEdge As SldWorks.Edge
    
    Dim SubAssyList As IArrListObject
    Set SubAssyList = New IArrListObject
    
    For i = LBound(vCompsIdx) To UBound(vCompsIdx)
    
        Dim xMin As Double
        Dim yMin As Double
        Dim xMax As Double
        Dim yMax As Double

        Dim oComp As IComp
        Set oComp = CompDict.Items(vCompsIdx(i))
            
        Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax, yMax + 16 * 0.0254, _
                                0, xMax, yMin - 16 * 0.0254, 0)
                                
        swSketchSegment.ConstructionGeometry = True
        
        Call AddSplitLineNote(swSketchSegment, swDrawing, swView)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp.GetComponent, xMax, swView, False, VisibleEdgesOnly)
        
        Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)
        
        Dim oSubAssy As ISubAssy
        Dim swDisplayDim As SldWorks.DisplayDimension
    
        If IsFrontView Then
            
            If i = LBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = 0
                oSubAssy.EndIdx = vCompsIdx(i)
                
                SubAssyList.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(NextAssyStartEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vOutline(1) - 0.015, swView)
                                
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = vCompsIdx(i - 1) + 1
                oSubAssy.EndIdx = vCompsIdx(i)
                
                SubAssyList.AddtoList oSubAssy

            End If
            
            Dim NextAssyComp As IComp
            Set NextAssyComp = CompDict.Items(vCompsIdx(i) + 1)
            
            Call GetViewMaxMinPoints(NextAssyComp, swView, xMin, xMax, yMin, yMax)
            Set NextAssyStartEdge = GetEdgeInView(NextAssyComp.GetComponent, xMin, swView, False, False)
            
            If i = UBound(vCompsIdx) Then
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swRightEdge, NextAssyStartEdge, swDrawing, _
                            NextAssyComp.xMax + 0.01, vOutline(1) - 0.015, swView)
                            
                Set oSubAssy.StartEdge = NextAssyStartEdge
                Set oSubAssy.EndEdge = swRightEdge
                Set oSubAssy.Dimension = swDisplayDim
                oSubAssy.AssyLength = swDisplayDim.GetDimension2(0).Value
                oSubAssy.StartIdx = vCompsIdx(i) + 1

                oSubAssy.EndIdx = (CompNoDict.Count) - 1
                
                SubAssyList.AddtoList oSubAssy
                            
            End If
            
        Else
            
            Dim TempComp As IComp
            Dim vPoint(2) As Double
            Dim vSheetPoint As Variant
    
            If i = LBound(vCompsIdx) Then
                
                Set TempComp = CompDict.Items(0)
                Call GetViewMaxMinPoints(TempComp, swView, xMin, xMax, yMin, yMax)
                vPoint(0) = xMin
                vPoint(1) = yMin
                vPoint(2) = 0
                
                vSheetPoint = GetSketchPointInSheetSpace(swView, vPoint)
                
                Set swLeftEdge = GetEdgeInView(TempComp.GetComponent, xMin, swView, False)
                
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swLeftEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView)
                                
                Set oSubAssy.StartEdge = swLeftEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                SubAssyList.AddtoList oSubAssy
                
            Else
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(SubAssyList.Items(UBound(SubAssyList.Items)).EndEdge, swEdge, swDrawing, _
                                oComp.xMin - 0.01, vSheetPoint(1) - 0.005, swView)
                                
                Set oSubAssy.StartEdge = SubAssyList.Items(UBound(SubAssyList.Items)).EndEdge
                Set oSubAssy.EndEdge = swEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                SubAssyList.AddtoList oSubAssy

            End If
            
            
            If i = UBound(vCompsIdx) Then
            
                Set TempComp = CompDict.Items(UBound(CompDict.Items))
                Call GetViewMaxMinPoints(TempComp, swView, xMin, xMax, yMin, yMax)
                Set swRightEdge = GetEdgeInView(TempComp.GetComponent, xMax, swView, False)
            
                Set oSubAssy = New ISubAssy
                Set swDisplayDim = SelectAndAddDimension(swEdge, swRightEdge, swDrawing, _
                            oComp.xMax + 0.01, vSheetPoint(1) - 0.005, swView)
                            
                Set oSubAssy.StartEdge = swEdge
                Set oSubAssy.EndEdge = swRightEdge
                Set oSubAssy.Dimension = swDisplayDim
                
                SubAssyList.AddtoList oSubAssy
                            
            End If
            
        End If
        
    Next i
    
    Set AddSplitLines = SubAssyList
    
End Function

Private Sub AddSplitLineNote(swSketchSegment As SldWorks.SketchLine, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    Dim swStartPoint As SldWorks.SketchPoint
    Set swStartPoint = swSketchSegment.GetStartPoint2
    
    swStartPoint.Select4 False, Nothing
    
    Dim vSketchPoint(2) As Double
    vSketchPoint(0) = swStartPoint.X
    vSketchPoint(1) = swStartPoint.Y
    vSketchPoint(2) = swStartPoint.Z
    
    
    Dim vPointInSheet As Variant
    vPointInSheet = GetSketchPointInSheetSpace(swView, vSketchPoint)
    
    Call AddNoteToView(swDrawing, "SPLIT LINE", vPointInSheet(0) + 0.005, vPointInSheet(1) + 0.00625)

End Sub

Private Function GetCompDictionary(FlatCompList As Variant, CompNoDict As Scripting.Dictionary) As Scripting.Dictionary
    
    Dim TempDict As New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(FlatCompList) To UBound(FlatCompList)
        
        TempDict.Add FlatCompList(i).GetComponent.Name2, FlatCompList(i)
        CompNoDict.Add FlatCompList(i).GetComponent.Name2, i
    
    Next i
    
    Set GetCompDictionary = TempDict
    
End Function

Function GetSelectedComponents() As Variant
    
    Dim swSelectionMgr As SldWorks.SelectionMgr
    Set swSelectionMgr = swTopLevelModel.SelectionManager
    
    Dim CompDict As Scripting.Dictionary
    Set CompDict = New Scripting.Dictionary
    
    If swSelectionMgr.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        For i = 0 To swSelectionMgr.GetSelectedObjectCount2(-1) - 1
            
            Dim swComp As SldWorks.Component2
            Set swComp = swSelectionMgr.GetSelectedObjectsComponent4(i + 1, -1)
            
            If False = CompDict.Exists(swComp.Name2) Then
                
                CompDict.Add swComp.Name2, swComp
            
            End If

        Next i
        
    End If
    
    If Not (CompDict.Count = 0) Then
    
        GetSelectedComponents = CompDict.Items
        
    End If

End Function
Private Sub ActivateDrawingDocument(swModel As SldWorks.ModelDoc2)
    
    Dim swFrame As SldWorks.Frame
    Set swFrame = swApp.Frame
    
    Dim vModelWindows As Variant
    vModelWindows = swFrame.ModelWindows
    
    Dim i As Integer
    For i = LBound(vModelWindows) To UBound(vModelWindows)
    
        Dim swModelWindow As SldWorks.ModelWindow
        Set swModelWindow = vModelWindows(i)
        
        If swModelWindow.Title = swModel.GetTitle Then
        
            swModelWindow.Activate
            Exit Sub
            
        End If
    
    Next i
End Sub

Private Function CheckForMultipleAssembly(ViewWidth As Double, ViewHeight As Double) As Boolean

    CheckForMultipleAssembly = False
    
    If ViewHeight <= 165 * 0.0254 Then
    
        If ViewWidth > 450 * 0.0254 Then
    
            CheckForMultipleAssembly = True
            
        End If
        
    Else
        
        If ViewWidth > 165 * 0.0254 Then
            
            CheckForMultipleAssembly = True
            
        End If
        
    End If

End Function

Private Sub AddLegendBlocks(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet)

    swDrawing.ActivateSheet swSheet.GetName
    
    Dim vSheetProp As Variant
    vSheetProp = swSheet.GetProperties
    
    Dim vPt(2) As Double
    vPt(0) = 0.1025 * vSheetProp(3)
    vPt(1) = 0.0161 * vSheetProp(3)
    vPt(2) = 0
    
    Dim SketchBlockInsertionPt As SldWorks.MathPoint
    Set SketchBlockInsertionPt = swMathUtility.CreatePoint(vPt)
    
    Dim swBlockDefinition As SldWorks.SketchBlockDefinition
    
'    If IsZChannelExists Then
    
         Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(SketchBlockInsertionPt, _
                "C:\FBD\COMMON\BLOCKS\MAKE UP+ASSEMBLY+ PART NUMBER LEGEND.SLDBLK", False, 0.02, 0)

'    Else
'
'        Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(Nothing, _
'                "C:\FBD\COMMON\BLOCKS\MAKEUP PANEL LEGEND FOR 6 SERIES.SLDBLK", False, vSheetProp(2) / vSheetProp(3), 0)
'
'    End If
    

    'Set swBlockInst = swDrawing.SketchManager.InsertSketchBlockInstance(swBlockDefinition, SketchBlockInsertionPt, 1, 0)

    
    swDrawing.GraphicsRedraw2

End Sub

Private Sub AddDimensionInFrontView(swView As SldWorks.View, FlatCompList As Variant, _
            DetailedCompList As Variant, swDrawing As SldWorks.ModelDoc2, MaxHeightComp As IComp, _
            ByRef swLeftEdge As SldWorks.Edge, ByRef swRightEdge As SldWorks.Edge)
            
    Dim vOutline As Variant
    vOutline = swView.GetOutline

    Dim LeftComp As IComp
    Set LeftComp = FlatCompList(0)
    
    Dim RightComp As IComp
    Set RightComp = FlatCompList(UBound(FlatCompList))
    
    Dim LeftxMin As Double
    Dim LeftxMax As Double
    Dim LeftYMin As Double
    Dim LeftYMax As Double
    Call GetViewMaxMinPoints(LeftComp, swView, LeftxMin, LeftxMax, LeftYMin, LeftYMax)
    
    Set swLeftEdge = GetEdgeInView(LeftComp.GetComponent, LeftxMin, swView, False)

    Dim RightxMin As Double
    Dim RightxMax As Double
    Dim RightYMin As Double
    Dim RightYMax As Double
    Call GetViewMaxMinPoints(RightComp, swView, RightxMin, RightxMax, RightYMin, RightYMax)
    
    
    Dim MaxCompXMin As Double
    Dim MaxCompXMax As Double
    Dim MaxCompYMin As Double
    Dim MaxCompXYMax As Double
    
    Call GetViewMaxMinPoints(MaxHeightComp, swView, MaxCompXMin, MaxCompXMax, MaxCompYMin, MaxCompXYMax)
    
    Set swRightEdge = GetEdgeInView(RightComp.GetComponent, RightxMax, swView, False)

    Dim swBottomDim As SldWorks.DisplayDimension
    Set swBottomDim = SelectAndAddDimension(swLeftEdge, swRightEdge, swDrawing, _
                                (vOutline(0) + vOutline(2)) / 2, vOutline(1) - 0.025, swView)
    
    Dim swBottomLeftEdge As SldWorks.Edge
    Set swBottomLeftEdge = GetEdgeInView(LeftComp.GetComponent, LeftYMin, swView, True)

    Dim swTopLeftEdge As SldWorks.Edge
    
'    Dim TotalArea As Double
    Dim swLeftDim As SldWorks.DisplayDimension

    If (Abs(LeftComp.yMax - RightComp.yMax) <= 0.5 * 0.0254 * swView.ScaleDecimal) Then
    
        Set swTopLeftEdge = GetEdgeInView(MaxHeightComp.GetComponent, MaxCompXYMax, swView, True)
        Set swLeftDim = SelectAndAddDimension(swTopLeftEdge, swBottomLeftEdge, _
                    swDrawing, vOutline(0) - 0.005, (vOutline(1) + vOutline(3)) / 2, swView)
        
'        If Not (swLeftDim Is Nothing) And Not (swBottomDim Is Nothing) Then
'
'            TotalArea = Round((swLeftDim.GetDimension2(0).Value * swBottomDim.GetDimension2(0).Value) / 144, 2)
'
'        End If
        
    Else
    
        Set swTopLeftEdge = GetEdgeInView(LeftComp.GetComponent, LeftYMax, swView, True)
        Set swLeftDim = SelectAndAddDimension(swTopLeftEdge, _
            swBottomLeftEdge, swDrawing, vOutline(0) - 0.005, (vOutline(1) + vOutline(3)) / 2, swView)
        
'        Dim LeftDimValue As Double
'        LeftDimValue = swLeftDim.GetDimension2(0).Value
        
        Dim swTopRightEdge As SldWorks.Edge
        Set swTopRightEdge = GetEdgeInView(RightComp.GetComponent, RightYMax, swView, True)

        Dim swBottomRightEdge As SldWorks.Edge
        Set swBottomRightEdge = GetEdgeInView(RightComp.GetComponent, RightYMin, swView, True)
        
        Dim swRightDim As SldWorks.DisplayDimension
        Set swRightDim = SelectAndAddDimension(swTopRightEdge, _
                        swBottomRightEdge, swDrawing, vOutline(2) + 0.005, (vOutline(1) + vOutline(3)) / 2, swView)
        
'        Dim RightDimValue As Double
'        RightDimValue = swRightDim.GetDimension2(0).Value
'
'        If RightDimValue > LeftDimValue Then
'
'            TotalArea = Round((RightDimValue * swBottomDim.GetDimension2(0).Value) / 144, 2)
'
'        Else
'
'            TotalArea = Round((LeftDimValue * swBottomDim.GetDimension2(0).Value) / 144, 2)
'
'        End If
        
    End If
    
    swBottomDim.SetText swDimensionTextParts_e.swDimensionTextCalloutBelow, "(XX.XX sq.ft)"

    
End Sub

Private Function SelectAndAddDimension(swEdge1 As SldWorks.Edge, swEdge2 As SldWorks.Edge, swDrawing As SldWorks.ModelDoc2, _
            xPos As Double, yPos As Double, swView As SldWorks.View) As SldWorks.DisplayDimension
    
    If Not (swEdge1 Is Nothing) And Not (swEdge2 Is Nothing) Then
    
        Call SelectEntity(swEdge1, False, swView)
        Call SelectEntity(swEdge2, True, swView)
        
        Set SelectAndAddDimension = swDrawing.AddHorizontalDimension2(xPos, yPos, 0)
        
        If Not SelectAndAddDimension Is Nothing Then
            SelectAndAddDimension.CenterText = True
            SelectAndAddDimension.SetDual2 False, False
            
        End If
    
    End If

End Function

Private Sub GetViewMaxMinPoints(oComp As IComp, swView As SldWorks.View, ByRef xMin As Double, _
                ByRef xMax As Double, ByRef yMin As Double, ByRef yMax As Double)

    Dim vViewMaxPt As Variant
    vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
    Dim vViewMinPt As Variant
    vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub

Private Sub AddStructuralNotes(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, Is12GAPanelExists)

    swDrawing.ActivateSheet swSheet.GetName
    
    Dim NoteCount As Integer
    NoteCount = 1
    
    Dim swStructuralNote As SldWorks.Note
    
    If Is12GAPanelExists Then

        Set swStructuralNote = swDrawing.CreateText2("<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. ALL CIRCLED PANELS ARE 12GA." & vbCrLf & _
         "2. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED RIBS." & vbCrLf & _
         "3. DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM OF CEILING PANELS, USE FOR CEILING L-ANGLE PLACEMENT.", 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)

    Else
        
        Set swStructuralNote = swDrawing.CreateText2("<FONT size=10PTS style=B> NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. RIB TO RIB #14 TEK SCREW @12" & Chr(34) & " O.C., UNLESS OTHERWISE SPECIFIED RIBS." & vbCrLf & _
         "2. DIMENSION FROM BOTTOM OF WALL PANEL TO BOTTOM OF CEILING PANELS, USE FOR CEILING L-ANGLE PLACEMENT.", 1.99241243641486E-02, 7.72464210842187E-02, 0, 0, 0)
         
     End If
    
    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft
End Sub

Private Sub InsertSketchBlock(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, ProjectNo As String)
    
    swDrawing.ActivateSheet swSheet.GetName
    
    Dim vSheetProp As Variant
    vSheetProp = swSheet.GetProperties
    
    Dim vPt(2) As Double
    vPt(0) = 0.01590679 * vSheetProp(3)
    vPt(1) = 0.00995866 * vSheetProp(3)
    vPt(2) = 0
    
    Dim SketchBlockInsertionPt As SldWorks.MathPoint
    Set SketchBlockInsertionPt = swMathUtility.CreatePoint(vPt)
    
    Dim swBlockDefinition As SldWorks.SketchBlockDefinition
    Set swBlockDefinition = swDrawing.SketchManager.MakeSketchBlockFromFile(SketchBlockInsertionPt, _
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " INTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)
                

End Sub

Private Sub UpdateBottomViewPosition(vComps As Variant, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Dim oComp As IComp
    Set oComp = vComps(0)
    
    Dim viewPosition As Variant
    viewPosition = swView.Position
    
    Dim wallPanelPosition As Variant
    wallPanelPosition = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
    
    If Abs(wallPanelPosition(1) - viewPosition(1)) / swView.ScaleDecimal > 5 * 0.0254 Then
    
        viewPosition(1) = viewPosition(1) + viewPosition(1) - wallPanelPosition(1)
        
    End If
    
    swView.Position = viewPosition
    
End Sub

Private Sub CleanUpActivateAndAddViewLabel(swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, wallName As String)

    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayOrigins, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayPlanes, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints2, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCurves, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayAllAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayCompAnnotations, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayReferencePoints, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLiveSections, False
    swDrawing.SetUserPreferenceToggle swUserPreferenceToggle_e.swDisplayLights, False
    
    swDrawing.ActivateView swView.Name
    
    Dim SheetDesc As String
    Dim LabelText As String
    If InStr(wallName, "Wall") > 0 Then
    
        SheetDesc = "STRUCTURAL, ELEVATION, EXTERNAL PANELS, " & UCase(wallName)
        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34) & _
         vbCrLf & "<FONT size=8PTS style=R> (INTERIOR VIEW)"
         
    Else
    
        SheetDesc = "STRUCTURAL, " & UCase(wallName)
        LabelText = "<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34)
        
    End If
    swDrawing.Extension.CustomPropertyManager("").Set2 "SHEET DESCRIPTION", SheetDesc
    swDrawing.Extension.CustomPropertyManager("").Set2 "ISSUED FOR", "CONSTRUCTION"
    
    Dim vOutline As Variant
    vOutline = swView.GetOutline
    
    Dim swLabelNote As SldWorks.Note

    Set swLabelNote = swDrawing.CreateText2("<FONT size=10PTS style=B> $PRP:" & Chr(34) & "SHEET DESCRIPTION" & Chr(34) & _
         vbCrLf & "<FONT size=8PTS style=R> (INTERIOR VIEW)", (vOutline(0) + vOutline(2)) / 2, vOutline(1) - 0.03, 0, 0, 0)
    swLabelNote.SetTextJustification swTextJustification_e.swTextJustificationCenter
    
    swDrawing.Extension.Rebuild swRebuildOptions_e.swCurrentSheetDisp

End Sub

Private Function Add12GACircles(vCompList As Variant, swDrawing As SldWorks.ModelDoc2, _
                swView As SldWorks.View) As Boolean
    
    Add12GACircles = False
    
    swDrawing.ActivateView swView.Name
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
    
    Dim i As Integer
    For i = LBound(vCompList) To UBound(vCompList)
    
        Dim oComp As IComp
        Set oComp = vCompList(i)

        Dim vViewMaxPt As Variant
        vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
        Dim vViewMinPt As Variant
        vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
            
        If oComp.GetCustomProperty("THK") = 0.1084 Then
            
            Add12GACircles = True

            Dim vPt(2) As Double
            vPt(0) = (vViewMaxPt(0) + vViewMinPt(0)) / 2
            vPt(1) = (vViewMaxPt(1) + vViewMinPt(1)) / 2
            vPt(2) = (vViewMaxPt(2) + vViewMinPt(2)) / 2
                
                
            Dim radius As Double
            radius = Sqr((vViewMaxPt(0) - vPt(0)) ^ 2 + (vViewMaxPt(1) - vPt(1)) ^ 2) + 0.0127
        
            Dim swSketchSegment As SldWorks.SketchSegment
            Set swSketchSegment = swSketchMgr.CreateCircleByRadius(vPt(0), vPt(1), vPt(2), radius)
            swSketchSegment.ConstructionGeometry = True
                
        End If
            
        If i = UBound(vCompList) Or i = LBound(vCompList) Then
                
            Call AddRibSketchAndNote(oComp, swView, swSketchMgr, swDrawing, i)
            
        End If

    Next i
    
    
    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
        
End Function

Private Sub AddRibSketchAndNote(oComp As IComp, swView As SldWorks.View, swSketchMgr As SldWorks.SketchManager, _
                swDrawing As SldWorks.DrawingDoc, CompPos As Integer)
    
    Dim Profile As String
    Profile = oComp.GetCustomProperty("Profile")
    
    Dim xMin As Double
    Dim yMin As Double
    Dim xMax As Double
    Dim yMax As Double
        
    Call GetViewMaxMinPoints(oComp, swView, xMin, xMax, yMin, yMax)
    
    Dim vSketchPoint As Variant
    
    If InStr(Profile, "CORNER") > 0 Then
    
        Dim swSketchSegmentHor As SldWorks.SketchSegment
        Dim swSketchSegmentVer As SldWorks.SketchSegment
    
        If InStr(Profile, "1") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 3.5, 1.5)
         
        ElseIf InStr(Profile, "2") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 1.5, 1.5)

        
        ElseIf InStr(Profile, "3") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 3.5, 3.5)
                
        
        ElseIf InStr(Profile, "4") > 0 Then
        
            Call CreateRibSketches(swSketchSegmentHor, swSketchSegmentVer, xMin, xMax, yMin, yMax, CompPos, swSketchMgr, 1.5, 3.5)
        
        End If
        
        If Not swSketchSegmentHor Is Nothing And Not swSketchSegmentVer Is Nothing Then
    
            Dim bool As Boolean
            bool = swDrawing.ActivateView(swView.Name)
            
            Call SelectSketchSegment(swSketchSegmentHor, swDrawing, swView, False, False)
            

            vSketchPoint = SelectSketchSegment(swSketchSegmentVer, swDrawing, swView, True, False)
            
            Call AddNoteToView(swDrawing, "RIB TO RIB" & vbCrLf & "#14 TEK SCREW" & vbCrLf & "@ 6" & Chr(34) & " O.C.", _
                            vSketchPoint(0) + 0.0075, vSketchPoint(1) + 0.0125)

        End If
    
    End If
    
    If Not (CompPos = 0) Then
        
        Dim swSketchSegment As SldWorks.SketchSegment
        Set swSketchSegment = swSketchMgr.CreateLine(xMax - 0.25 * 0.0254, yMin, _
                                0, xMax + 16 * 0.0254, yMin, 0)
                                
        swSketchSegment.ConstructionGeometry = True
        
        vSketchPoint = SelectSketchSegment(swSketchSegment, swDrawing, swView, False, True)
        Call AddNoteToView(swDrawing, "CASTING BED", vSketchPoint(0) + 0.0075, vSketchPoint(1) - 0.005)
        
        Dim swEdge As SldWorks.Edge
        Set swEdge = GetEdgeInView(oComp.GetComponent, yMin, swView, True)
        
        Call AddCollinearRelation(swDrawing, swEdge, swSketchSegment, swView)

    End If
    
End Sub

Private Sub AddCollinearRelation(swDrawing As SldWorks.DrawingDoc, swEdge As SldWorks.Edge, swSketchSegment As SldWorks.SketchSegment, swView As SldWorks.View)
    
    If Not (swEdge Is Nothing) And Not (swSketchSegment Is Nothing) Then
    
        Call SelectEntity(swEdge, False, swView)
        swSketchSegment.Select4 True, Nothing
                
        swDrawing.SketchAddConstraints "sgCOLINEAR"
        
    End If
    
End Sub

Function GetEdgeInView(swComp As SldWorks.Component2, ValToMatch As Double, swView As SldWorks.View, _
    IsHorizontal As Boolean, Optional CheckAllVisibleEdgesOnly As Boolean = True) As SldWorks.Edge
    
    Dim idx As Integer
    If IsHorizontal Then
        
        idx = 1
    Else
    
        idx = 0
        
    End If

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
            IsSelected = SelectEntity(swEdge, False, swView)
            
            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsLine Then
            
                Dim vStartPoint As Variant
                vStartPoint = swEdge.GetStartVertex.GetPoint
                vStartPoint = GetComponentPointInViewSpace(swComp, vStartPoint, swView)
                
                Dim vEndPoint As Variant
                vEndPoint = swEdge.GetEndVertex.GetPoint
                vEndPoint = GetComponentPointInViewSpace(swComp, vEndPoint, swView)
                
                If Abs(vStartPoint(idx) - vEndPoint(idx)) <= 0.00001 And Abs(vStartPoint(idx) - ValToMatch) <= 0.00001 Then

                    Set GetEdgeInView = swEdge
                    Exit Function
                    
                End If
            
            End If
            
        Next i

    End If

End Function


Sub AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double)
            
    Dim swNote As SldWorks.Note
    Set swNote = swDrawing.InsertNote(NoteText)
            
    If Not swNote Is Nothing Then

        Dim swAnnotation As SldWorks.Annotation
        Set swAnnotation = swNote.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
End Sub

Sub CreateRibSketches(ByRef swSketchSegmentHor As SldWorks.SketchSegment, ByRef swSketchSegmentVer As SldWorks.SketchSegment, _
                    xMin As Double, xMax As Double, yMin As Double, yMax As Double, CompPos As Integer, swSketchMgr As SldWorks.SketchManager, _
                        OffSetVer As Double, OffSetHor As Double)
                    
    Const Length As Double = 3
    Const FrontOffset As Double = 1.5

    If CompPos = 0 Then
    
        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMax - OffSetHor * 0.0254, yMin + FrontOffset * 0.0254, _
                                0, xMax - (OffSetHor - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)
                                
        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMin + FrontOffset * 0.0254, yMax - OffSetVer * 0.0254, _
                                0, xMin + FrontOffset * 0.0254, yMax - (OffSetVer - Length) * 0.0254, 0)
                                
    Else

        Set swSketchSegmentHor = swSketchMgr.CreateLine(xMin + OffSetVer * 0.0254, yMin + FrontOffset * 0.0254, _
                                0, xMin + (OffSetVer - Length) * 0.0254, yMin + FrontOffset * 0.0254, 0)

        Set swSketchSegmentVer = swSketchMgr.CreateLine(xMax - FrontOffset * 0.0254, yMax - OffSetHor * 0.0254, _
                                0, xMax - FrontOffset * 0.0254, yMax - (OffSetHor - Length) * 0.0254, 0)

    End If
    
End Sub

Function SelectSketchSegment(swSketchSegment As SldWorks.SketchSegment, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, Append As Boolean, IsSelectMid As Boolean)
    
    Dim swSketchLine As SldWorks.SketchLine
    Set swSketchLine = swSketchSegment
    
    Dim swStartPoint As SldWorks.SketchPoint
    Set swStartPoint = swSketchLine.GetStartPoint2
    
    Dim swEndPoint As SldWorks.SketchPoint
    Set swEndPoint = swSketchLine.GetEndPoint2
    
    Dim vSketchPoint(2) As Double
    If IsSelectMid Then
    
        vSketchPoint(0) = (swStartPoint.X + swEndPoint.X) / 2
        vSketchPoint(1) = (swStartPoint.Y + swEndPoint.Y) / 2
        vSketchPoint(2) = (swStartPoint.Z + swEndPoint.Z) / 2
    Else
    
        vSketchPoint(0) = swEndPoint.X
        vSketchPoint(1) = swEndPoint.Y
        vSketchPoint(2) = swEndPoint.Z
        
    End If
    
    Dim vPointInSheet As Variant
    vPointInSheet = GetSketchPointInSheetSpace(swView, vSketchPoint)
    
    swDrawing.Extension.SelectByID2 "Line" & swSketchSegment.GetID(1), "SKETCHSEGMENT", vPointInSheet(0), vPointInSheet(1), vPointInSheet(2), Append, -1, Nothing, 0
    SelectSketchSegment = vPointInSheet
    
End Function



Sub UpdateHatchProperties(swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim vSketchHatches As Variant
    vSketchHatches = swSketch.GetSketchHatches
            
    If Not IsEmpty(vSketchHatches) Then
            
        Dim i As Integer
        For i = LBound(vSketchHatches) To UBound(vSketchHatches)
                
            Dim swSketchHatch As SldWorks.SketchHatch
            Set swSketchHatch = vSketchHatches(i)
                    
            swSketchHatch.Pattern = "ANSI32 (Steel)"
            swSketchHatch.Scale2 = swView.ScaleDecimal * 3
                
        Next i
                
    End If

End Sub

Private Sub AddCallouts(vConsolidatedList As Variant, swDrawing As SldWorks.ModelDoc2, swView As SldWorks.View, _
        MaxCompHeight As Double, ByRef IsMakeUpExists As Boolean)
    
    Const SheetPosForLastBalloon As Double = 0.2655
    Const Increment As Double = 0.005
    Const MaxBalloonWidth As Double = 0.015875
    
    IsMakeUpExists = False
    
    swDrawing.Extension.SetUserPreferenceInteger swUserPreferenceIntegerValue_e.swDetailingBOMUpperText, swUserPreferenceOption_e.swDetailingNoOptionSpecified, swBalloonTextContent_e.swBalloonTextPartNumberBOM
    
    Dim maxNoOfBalloons As Integer
    maxNoOfBalloons = Int((SheetPosForLastBalloon - MaxCompHeight) / Increment)
    
    Dim AddorSub As Integer
    AddorSub = 1
    
    Dim BalloonCount As Integer
    BalloonCount = 1
    
    Dim AnnXPos As Double
    Dim AnnYPos As Double
    
    Dim i As Integer
    For i = LBound(vConsolidatedList) To UBound(vConsolidatedList)
    
        Dim oList As IConsolidatedList
        Set oList = vConsolidatedList(i)
        
        Dim oComp As IComp
        Set oComp = oList.Comp

        swDrawing.ClearSelection2 True

        Dim xPos As Double
        Dim yPos As Double
      
        xPos = (oComp.xMin + oComp.xMax) / 2 - Abs((oComp.xMin - oComp.xMax) / 2) + 3.5 * 0.0254 * swView.ScaleDecimal
        yPos = 0.075 * oComp.yMin + 0.925 * oComp.yMax
        
        
        If oComp.IsTop Then
        
            If Not (i = LBound(vConsolidatedList)) Then
    
                Dim prevComp As IComp
                Set prevComp = vConsolidatedList(i - 1).Comp
    
                If AddorSub = -1 Then
    
                    If Abs(prevComp.xMin - oComp.xMin) > 2 * MaxBalloonWidth Then
    
                        AddorSub = 1
                        BalloonCount = 1
    
                    End If
    
                Else
    
                    If Abs(prevComp.xMin - oComp.xMin) > MaxBalloonWidth Then
    
                        AddorSub = 1
                        BalloonCount = 1
    
                    End If
    
                End If
                
            End If
            
            If AddorSub = 1 Then
            
                If BalloonCount > maxNoOfBalloons Then
                    
                    AddorSub = -1
                    BalloonCount = BalloonCount + AddorSub
                    
                End If
            
            Else
            
                If BalloonCount < 1 Then
                
                    xPos = (oComp.xMin + oComp.xMax) / 2 + Abs((oComp.xMin - oComp.xMax) / 2) - 3.5 * 0.0254 * swView.ScaleDecimal
                    BalloonCount = maxNoOfBalloons
                    
                End If
                
            End If
            
            AnnXPos = xPos
            AnnYPos = MaxCompHeight + BalloonCount * Increment
            BalloonCount = BalloonCount + AddorSub
            
        ElseIf oComp.IsBottom Then
        
            xPos = (oComp.xMin + oComp.xMax) / 2
            yPos = 0.7 * oComp.yMin + 0.3 * oComp.yMax
            AnnXPos = xPos
            AnnYPos = oComp.yMin - Increment
            
        Else
        
            xPos = (oComp.xMin + oComp.xMax) / 2
            yPos = 0.3 * oComp.yMin + 0.7 * oComp.yMax
            AnnXPos = oComp.xMin - 3 * Increment
            AnnYPos = yPos - 2 * Increment
            
        End If
       
    
        Dim IsSelected As Boolean
        IsSelected = False
        Call SelectComponent(swDrawing, oComp, xPos, yPos, 1, IsSelected, swView)
        
        If IsSelected Then
        
            Dim swBalloonParams As SldWorks.BalloonOptions
            Set swBalloonParams = swDrawing.Extension.CreateBalloonOptions()
            swBalloonParams.Size = swBalloonFit_e.swBF_Tightest
            swBalloonParams.Style = swBalloonStyle_e.swBS_Inspection
            
            If oList.Qty > 1 Then
    
                swBalloonParams.ShowQuantity = True
                swBalloonParams.QuantityOverride = True
                swBalloonParams.QuantityOverrideValue = CStr(oList.Qty)
                
            End If
            
            Dim swComp As SldWorks.Component2
            Set swComp = oComp.GetComponent
            'Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))
            
            Dim swNote As SldWorks.Note
            Set swNote = swDrawing.Extension.InsertBOMBalloon2(swBalloonParams)
            
            If Not swNote Is Nothing Then

                Dim swAnn As SldWorks.Annotation
                Set swAnn = swNote.GetAnnotation
                swAnn.SetPosition2 AnnXPos, AnnYPos, 0
                
                If oComp.IsTop Then
                
                    If AddorSub = 1 Then
                    
                        Dim vNoteExtents As Variant
                        vNoteExtents = swNote.GetExtent
     
                        If oList.Qty > 1 Then
                        
                             AnnXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0064
                        
                        Else
                            AnnXPos = xPos - ((vNoteExtents(3) - vNoteExtents(0))) + 0.0027
                            
                        End If
            
                        swAnn.SetPosition2 AnnXPos, AnnYPos, 0
    
                    End If
                    
                End If
                
            End If
            
        End If
        
        Call AddHatchForMakeUpPanel(oComp, swDrawing, IsMakeUpExists, swView)

    Next i
    
    Call UpdateHatchProperties(swView)

End Sub

Private Sub AddHatchForMakeUpPanel(oComp As IComp, swDrawing As SldWorks.ModelDoc2, ByRef IsMakeUpExists As Boolean, swView As SldWorks.View)
    
    swDrawing.ClearSelection2 True
    
    If oComp.IsMakeUpPanel Then
    
        IsMakeUpExists = True
            
        Dim vFaces As Variant
        vFaces = oComp.GetViewNormalFaces
        
        Dim i As Integer
        For i = LBound(vFaces) To UBound(vFaces)
            
            Dim swFace As SldWorks.Face2
            Set swFace = vFaces(i)
            Call SelectEntity(swFace, True, swView)
            swDrawing.InsertHatchedFace
            
        Next i
            
    End If

End Sub

Private Sub SelectComponent(swDrawing As SldWorks.ModelDoc2, oComp As IComp, xPos As Double, _
    yPos As Double, Count As Integer, IsSelected As Boolean, swView As SldWorks.View)
    
    IsSelected = swDrawing.Extension.SelectByID2("", "FACE", xPos, yPos, _
                    0, False, -1, Nothing, 1)
                    
    If Count > 2 Then
        
        Dim vFaces As Variant
        vFaces = oComp.GetViewNormalFaces
        
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(0)
        IsSelected = SelectEntity(swFace, False, swView)
        Exit Sub
        
    End If
    
    If IsSelected Then
    
        Dim swSelectMgr As SldWorks.SelectionMgr
        Set swSelectMgr = swDrawing.SelectionManager
        
        Dim swComp As SldWorks.DrawingComponent
        Set swComp = swSelectMgr.GetSelectedObjectsComponent4(2, -1)
        
        If Not (Right(swComp.Name, Len(swComp.Name) - InStrRev(swComp.Name, "/")) = _
            Right(oComp.GetComponent.Name2, Len(oComp.GetComponent.Name2) - InStrRev(oComp.GetComponent.Name2, "/"))) Then
            
            Call SelectComponent(swDrawing, oComp, (oComp.xMax + oComp.xMin) / 2, yPos, Count + 1, IsSelected, swView)
            
        End If
        
    Else
    
        Call SelectComponent(swDrawing, oComp, (oComp.xMax + oComp.xMin) / 2, yPos, Count + 1, IsSelected, swView)
        
    End If
    
    
End Sub

Function GetViewName(ByRef wallName As String)

    Select Case wallName
        
        Case "Wall-A"
            
            GetViewName = "*Back"
        
        Case "Wall-B"
            
            GetViewName = "*Right"
        
        Case "Wall-C"
        
            GetViewName = "*Front"
        
        Case "Wall-D"
            
            GetViewName = "*Left"
            
        Case "Ceiling"
            
            GetViewName = "*Top"
            
        Case Else
            
            ViewNameForm.Show
            wallName = ViewNameForm.WallNameBox.Value
            GetViewName = ViewNameForm.ViewNameBox.Value
            Unload ViewNameForm
    
    End Select
    
End Function

Function GetViewVector(viewName As String) As Double()

    Select Case viewName
        
        Case "*Back"
            
            GetViewVector = GetOppositeVector(zDirectionVector)
        
        Case "*Right"

            GetViewVector = xDirectionVector
        
        Case "*Front"
        
            GetViewVector = zDirectionVector
        
        Case "*Left"
            
            GetViewVector = GetOppositeVector(xDirectionVector)
            
        Case "*Top"
            
            GetViewVector = yDirectionVector
            
        Case "*Bottom"
            
            GetViewVector = GetOppositeVector(yDirectionVector)
    
    End Select
       
End Function

Function ScaleAndInsertBottomView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
            

    Dim xScale As Integer
    Dim yScale As Integer
    xScale = GetScaleValue(swView.ScaleDecimal * 0.40005 / ViewWidth)
    yScale = GetScaleValue(swView.ScaleDecimal * 0.1574625 / ViewHeight) '0.20995
    
    Dim IsScaleSet As Boolean
    IsScaleSet = False
    
    If xScale > 0 And yScale > 0 Then
        
        If yScale > xScale Then
            
            IsScaleSet = swView.Sheet.SetScale(1, yScale, True, True)
           
        Else
            
            IsScaleSet = swView.Sheet.SetScale(1, xScale, True, True)
        
        End If
        
    End If
    
    Dim IsViewSelected As Boolean
    
    Dim swDrawingModel As SldWorks.ModelDoc2
    Set swDrawingModel = swDrawing
    
    IsViewSelected = swDrawingModel.Extension.SelectByID2(swView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, 0)
    Set ScaleAndInsertBottomView = swDrawing.CreateUnfoldedViewAt3(0.21593179, 0.08695241, 0, False)

End Function

Function GetScaleValue(scaleVal As Double) As Integer

    GetScaleValue = 0
    
    Dim stdScales As Variant
    stdScales = Array(1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
    
    Dim i As Integer
    For i = LBound(stdScales) To UBound(stdScales)
    
        If stdScales(i) >= (1 / scaleVal) Then
           GetScaleValue = stdScales(i)
           Exit For
        End If
    
    Next i

End Function

Function GetComponentsSortedWithYPosition(swView As SldWorks.View, swDrawing As SldWorks.ModelDoc2, _
            swViewNormalVector As SldWorks.MathVector, ByRef ViewWidth As Double, ByRef ViewHeight As Double, _
                ByRef MaxHeightComp As IComp, ByRef IsZChannelExists As Boolean) As IArrListObject
    
    swDrawing.ActivateView swView.Name
    
    Dim vComps As Variant
    vComps = swView.GetVisibleComponents()
    
    IsZChannelExists = False
    
    Dim swTopLevelAssy As SldWorks.AssemblyDoc
    Set swTopLevelAssy = swTopLevelModel

    Dim CompList As IArrListObject
    Set CompList = New IArrListObject

    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)

        Dim swCompFromRoot As SldWorks.Component2
        Set swCompFromRoot = swTopLevelAssy.GetComponentByName(Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/")))
        
        If swCompFromRoot.GetSuppression = swComponentSuppressionState_e.swComponentLightweight Then
            
            Dim bRet As Integer
            bRet = swCompFromRoot.SetSuppression2(swComponentSuppressionState_e.swComponentResolved)
            
        End If

        Dim swCompModel As SldWorks.ModelDoc2
        Set swCompModel = swCompFromRoot.GetModelDoc2

        If Not swCompModel Is Nothing Then
            
            Dim swCompProp As SldWorks.CustomPropertyManager
            Set swCompProp = swCompModel.Extension.CustomPropertyManager("")
            
            Dim Profile As String
            Dim ResolvedVal As String
            Dim wasResolved As Boolean
            swCompProp.Get5 "Profile", False, Profile, ResolvedVal, wasResolved
            
            If InStr(Profile, "EXT-") > 0 Then
            
                CompList.AddtoList GetComponentWithPosition(swCompFromRoot, swView, swCompModel, swDrawing, swViewNormalVector)
            
            ElseIf InStr(Profile, "Z-CHANNEL") > 0 Then
                
                IsZChannelExists = True
                
            End If
        
        End If
        
    Next i
    
    CompList.SortItems "yMin", False
    
    Dim MinHeight As Double
    MinHeight = CompList.Items(LBound(CompList.Items)).yMin
    
    CompList.SortItems "xMin", False
    ViewWidth = CompList.Items(UBound(CompList.Items)).xMax - CompList.Items(LBound(CompList.Items)).xMin

    CompList.SortItems "yMax"
    Set MaxHeightComp = CompList.Items(LBound(CompList.Items))
    ViewHeight = MaxHeightComp.yMax - MinHeight
    
    Set GetComponentsSortedWithYPosition = CompList

End Function

Function GetComponentWithPosition(swComp As SldWorks.Component2, swView As SldWorks.View, _
        swCompModel As SldWorks.ModelDoc2, swDrawing As SldWorks.ModelDoc2, _
        swViewNormalVector As SldWorks.MathVector) As IComp

    Dim vFaces As Variant
    vFaces = GetComponentFaces(swComp) 'swView.GetVisibleEntities2(swComp, swViewEntityType_e.swViewEntityType_Face)
    
    Dim vBodies As Variant
    vBodies = swCompModel.GetBodies(swSolidBody)

    
    Dim swBody As SldWorks.Body2
    Set swBody = vBodies(0) 'vEnts(0).GetBody

    If Not IsEmpty(vFaces) Then
    
        Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))

        Dim vNormalFaces As Variant
        vNormalFaces = GetNormalFaces(vFaces, swComp.Transform2, swViewNormalVector)

        Dim vBodyBounds As Variant
        vBodyBounds = swBody.GetBodyBox
            
        Dim vBodyMinPoint(2) As Double
        Dim vBodyMaxPoint(2) As Double
            
        vBodyMinPoint(0) = vBodyBounds(0)
        vBodyMinPoint(1) = vBodyBounds(1)
        vBodyMinPoint(2) = vBodyBounds(2)
            
        vBodyMaxPoint(0) = vBodyBounds(3)
        vBodyMaxPoint(1) = vBodyBounds(4)
        vBodyMaxPoint(2) = vBodyBounds(5)
            
        Dim MinPoint As Variant
        MinPoint = GetComponentPointInSheetSpace(swComp, vBodyMinPoint, swView)

        Dim MaxPoint As Variant
        MaxPoint = GetComponentPointInSheetSpace(swComp, vBodyMaxPoint, swView)
            
        Dim oComp As IComp
        Set oComp = New IComp
        oComp.Initialize swComp, MinPoint, MaxPoint, vBodyMinPoint, vBodyMaxPoint, vNormalFaces
        
    End If
    
    Debug.Print Right(swComp.Name2, Len(swComp.Name2) - InStrRev(swComp.Name2, "/"))
    
    Set GetComponentWithPosition = oComp

End Function

Function GetComponentFaces(swComp As SldWorks.Component2)
    
    Dim TempFaces As Variant
    
    Dim vBodies As Variant
    vBodies = swComp.GetBodies3(swBodyType_e.swSolidBody, swBodyInfo_e.swNormalBody_e)
    
    Dim i As Integer
    Dim j As Integer
    For i = LBound(vBodies) To UBound(vBodies)
    
        Dim swBody As SldWorks.Body2
        Set swBody = vBodies(i)
        
        Dim vFaces As Variant
        vFaces = swBody.GetFaces
        
        If i = 0 Then
            
            TempFaces = vFaces
            
        Else
            
            Call CombineArr(TempFaces, vFaces)
            
        End If
    
    Next i
    
    GetComponentFaces = TempFaces

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
            
            Call CombineArr(TempEdges, vEdges)
            
        End If
    
    Next i
    
    GetComponentEdges = TempEdges

End Function

Function CombineArr(ByRef MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
End Function

Function GetNormalFaces(vFaces As Variant, CompTransform As IMathTransform, _
    swViewNormalVector As SldWorks.MathVector) As Variant
    
    Dim FaceCount As Integer
    FaceCount = 0
    
    Dim NormalFaces() As SldWorks.Face2

    Dim i As Integer
    For i = LBound(vFaces) To UBound(vFaces)

        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)

        Dim swSurface As SldWorks.Surface
        Set swSurface = swFace.GetSurface
        
        Dim swFaceNormalVector As SldWorks.MathVector
        Set swFaceNormalVector = swMathUtility.CreateVector(swFace.Normal)
        
        Set swFaceNormalVector = swFaceNormalVector.MultiplyTransform(CompTransform)
        Set swFaceNormalVector = swFaceNormalVector.Normalise
        
        Dim Angle As Double
        Angle = Arccos(swFaceNormalVector.Dot(swViewNormalVector)) * 180# / 3.14159265359
 
        If swSurface.IsPlane And Angle <= 0.01 Then
            
            Dim swEnt As SldWorks.Entity
            Set swEnt = swFace
            Set swEnt = swEnt.GetSafeEntity
            
            ReDim Preserve NormalFaces(FaceCount)
            Set NormalFaces(FaceCount) = swEnt
            FaceCount = FaceCount + 1
            
        End If

    Next i
    
    GetNormalFaces = NormalFaces

End Function

Private Function GetComponentsSortedWithXPosition(vComps As Variant, ByRef FlatCompList As Variant, _
        swView As SldWorks.View, ByRef MaxCompHeight As Double) As Variant

    Dim CompWithPosDict As Scripting.Dictionary
    Set CompWithPosDict = New Scripting.Dictionary
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim oComp As IComp
        Set oComp = vComps(i)

        Dim MinPoint As Variant
        MinPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
        Dim MaxPoint As Variant
        MaxPoint = GetComponentPointInSheetSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
        
        oComp.UpdateSheetMaxMinDimensions swView, MinPoint, MaxPoint
        
        If i = LBound(vComps) Then
            
            MaxCompHeight = oComp.yMax
        
        End If

        Dim keyPosVal As Double
        keyPosVal = Round(oComp.xMin, 4)
        
        If CompWithPosDict.Exists(keyPosVal) Then
            
            Dim ExistingArr As Variant
            ExistingArr = CompWithPosDict(keyPosVal)
            
            ReDim Preserve ExistingArr(UBound(ExistingArr) + 1)
            Set ExistingArr(UBound(ExistingArr)) = oComp
            
            CompWithPosDict(keyPosVal) = ExistingArr
            
        Else
            
            Dim TempArr(0) As IComp
            Set TempArr(0) = oComp
            
            CompWithPosDict.Add keyPosVal, TempArr
            
        End If

    Next i

    GetComponentsSortedWithXPosition = GetDetailedCompList(CompWithPosDict, FlatCompList)
    
End Function

Private Function GetDetailedCompList(CompWithPosDict As Scripting.Dictionary, ByRef FlatCompList) As Variant

    Dim keysArrList As IArrList
    Set keysArrList = New IArrList
    
    keysArrList.AddItems = CompWithPosDict.Keys
    keysArrList.SortItems False
    
    Dim vItemsPos As Variant
    vItemsPos = keysArrList.Items
    
    Dim TempArr() As Variant
    
    Dim i As Integer
    For i = LBound(vItemsPos) To UBound(vItemsPos)
        
        ReDim Preserve TempArr(i)
        TempArr(i) = CompWithPosDict(vItemsPos(i))

        If i = 0 Then
                
               FlatCompList = TempArr(i)
                
        Else
            
            Call CombineArr(FlatCompList, TempArr(i))
            
        End If

    Next i
    
    GetDetailedCompList = TempArr
    
End Function

Private Function GetConsolidatedList(vCompsOfComps As Variant, ByRef DoorList As IArrListObject, ByRef HVACList As IArrListObject) As Variant

    Dim vConsolidatedLists As Variant
    Dim List As IConsolidatedList

    Dim IsInit As Boolean
    IsInit = True

    Dim HVACSubAssy As IDoorOrHVACAssy
    
    Dim IsHVACStarted As Boolean
    IsHVACStarted = False
    
    Dim DoorSubAssy As IDoorOrHVACAssy
    
    Dim IsDoorStarted As Boolean
    IsDoorStarted = False
    
    Dim StartIndex As Integer
    Dim EndIndex As Integer
    
    Dim LastComp As IComp
        
    Dim i As Integer
    
    For i = LBound(vCompsOfComps) To UBound(vCompsOfComps)
        
        Dim vComps As Variant
        vComps = vCompsOfComps(i)
        
        If UBound(vComps) = 0 Then
        
            If IsHVACStarted Then
            
                IsHVACStarted = False
                Set HVACSubAssy.EndComp = vComps(0)
                EndIndex = i - 1
                HVACList.AddtoList HVACSubAssy
                Call UpdatedConsolidatedList(vConsolidatedLists, IsInit, StartIndex, EndIndex, vCompsOfComps)
                
            End If
        
            Dim oComp As IComp
            Set oComp = vComps(0)
            
            If IsInit Then
                
                Set List = New IConsolidatedList
                Set List.Comp = oComp
    
                ReDim vConsolidatedLists(0)
                Set vConsolidatedLists(0) = List
                
                IsInit = False
                
            Else
                
                Dim LastConsolidatedItem As IConsolidatedList
                Set LastConsolidatedItem = vConsolidatedLists(UBound(vConsolidatedLists))

                If LastConsolidatedItem.Comp.GetPathName = oComp.GetPathName Then

                    LastConsolidatedItem.IncQty

                Else

                    Set List = New IConsolidatedList
                    Set List.Comp = oComp
    
                    ReDim Preserve vConsolidatedLists(UBound(vConsolidatedLists) + 1)
                    Set vConsolidatedLists(UBound(vConsolidatedLists)) = List
    
                End If
                
            End If
            
            If Not LastComp Is Nothing Then
            
                If Not Abs(LastComp.yMin - oComp.yMin) <= 0.00001 Then

                    If IsDoorStarted Then
                
                        IsDoorStarted = False
                        Set DoorSubAssy.EndComp = oComp
                        DoorSubAssy.DoororHVACWidth = oComp.xMin - DoorSubAssy.StartComp.xMax
                        DoorSubAssy.DoororHVACLength = Abs(LastComp.yMin - oComp.yMin)
                        
                        DoorList.AddtoList DoorSubAssy
                        
                    Else
                    
                        IsDoorStarted = True
                        Set DoorSubAssy = New IDoorOrHVACAssy
                        Set DoorSubAssy.StartComp = LastComp

                    End If
                    
                End If
                
            End If
        
            Set LastComp = vComps(0)

        Else
            
            If False = IsHVACStarted Then
                
                IsHVACStarted = True
                StartIndex = i
                Set HVACSubAssy = New IDoorOrHVACAssy
                Set HVACSubAssy.StartComp = vCompsOfComps(i - 1)(0)
                
            End If

        End If

    Next i

    GetConsolidatedList = vConsolidatedLists

End Function

Private Sub UpdatedConsolidatedList(ByRef vConsolidatedLists As Variant, IsInit As Boolean, StartIndex As Integer, _
                EndIndex As Integer, vCompsOfComps As Variant)
                
    Dim i As Integer
    
    Dim TempDictComp As Scripting.Dictionary
    Set TempDictComp = New Scripting.Dictionary
    
    Dim TempDictQty As Scripting.Dictionary
    Set TempDictQty = New Scripting.Dictionary
    
    For i = StartIndex To EndIndex
    
        Dim vComps As Variant
        vComps = vCompsOfComps(i)
        
        Dim j As Integer
        For j = LBound(vComps) To UBound(vComps)
        
            Dim oComp As IComp
            Set oComp = vComps(j)
            
            If Not (j = LBound(vComps)) Then
                
                oComp.IsTop = False
            
            End If
            
            If j = UBound(vComps) Then
                
                oComp.IsBottom = True
                
            End If
            
            Dim PathVal As String
            PathVal = oComp.GetComponent.GetPathName
            
            If TempDictComp.Exists(PathVal) Then
            
                TempDictQty.Item(PathVal) = TempDictQty.Item(PathVal) + 1
                
            Else
            
                TempDictComp.Add PathVal, oComp
                TempDictQty.Add PathVal, 1
                
            End If
        
        Next j
        
    Next i
    
    Dim DictKeys As Variant
    DictKeys = TempDictComp.Keys
    
    Dim k As Integer
    For k = LBound(DictKeys) To UBound(DictKeys)
    
        Dim List As IConsolidatedList
        Set List = New IConsolidatedList
        Set List.Comp = TempDictComp.Item(DictKeys(k))
        List.Qty = TempDictQty.Item(DictKeys(k))
        
        If IsInit Then

            ReDim vConsolidatedLists(0)
            Set vConsolidatedLists(0) = List
                
            IsInit = False
            
        Else
        
            ReDim Preserve vConsolidatedLists(UBound(vConsolidatedLists) + 1)
            Set vConsolidatedLists(UBound(vConsolidatedLists)) = List
        
        End If

    Next k

End Sub

Function GetComponentPointInSheetSpace(swComp As SldWorks.Component2, _
                vPoint As Variant, swView As SldWorks.View)
    
    GetComponentPointInSheetSpace = GetTransformPoint(vPoint, _
                                swComp.Transform2.Multiply(swView.ModelToViewTransform))

End Function

Function GetTransformPoint(vPoint As Variant, swTransform As SldWorks.MathTransform)
    
    Dim swMathPoint As SldWorks.MathPoint
    Set swMathPoint = swMathUtility.CreatePoint(vPoint)
    
    Set swMathPoint = swMathPoint.MultiplyTransform(swTransform)
    GetTransformPoint = swMathPoint.ArrayData

End Function

Private Function GetSketchPointInSheetSpace(swView As SldWorks.View, vPoint As Variant)

    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    GetSketchPointInSheetSpace = GetTransformPoint(vPoint, swSketch.ModelToSketchTransform.Inverse)

End Function

Function GetComponentPointInViewSpace(swComp As SldWorks.Component2, _
                    vPoint As Variant, swView As SldWorks.View)
    
    Dim swSketch As SldWorks.Sketch
    Set swSketch = swView.GetSketch
    
    Dim XForm As SldWorks.MathTransform
    Set XForm = swComp.Transform2.Multiply(swView.ModelToViewTransform)
    Set XForm = XForm.Multiply(swSketch.ModelToSketchTransform)
    
    GetComponentPointInViewSpace = GetTransformPoint(vPoint, XForm)

End Function


