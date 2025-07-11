VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} HideShowForm 
   Caption         =   "Hide/ Show Components"
   ClientHeight    =   9108.001
   ClientLeft      =   108
   ClientTop       =   456
   ClientWidth     =   6636
   OleObjectBlob   =   "HideShowForm.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "HideShowForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False

Option Explicit

Dim swSketchMgr As SldWorks.SketchManager
Const BalloonWidth As Double = 0.0065
Const SheetBorderTop As Double = 0.27030866
Const SheetBorderLeft As Double = 0.01590679
Const SheetBorderRight As Double = 0.41595679

Dim RebarCompDict As Scripting.Dictionary
Dim FoamCompDict As Scripting.Dictionary

Dim xDirectionVector(2) As Double
Dim yDirectionVector(2) As Double
Dim zDirectionVector(2) As Double

Const HorizontalMaxDim As Double = 0.371
Const VerticalMaxDim As Double = 0.1295

Private Sub AddEpsFoamButton_Click()
    
    Call AddToSelectionBox(Me.EpsFoamListBox, FoamCompDict)
    
End Sub

Private Sub AddRebarButton_Click()

    Call AddToSelectionBox(Me.RebarListBox, RebarCompDict)
    
End Sub

Private Sub AddToSelectionBox(SelListBox As MSForms.ListBox, ByRef Dict As Scripting.Dictionary)

    Dim swModel As SldWorks.ModelDoc2
    Set swModel = swApp.ActiveDoc
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) > 0 Then
    
        Dim i As Integer
        For i = 1 To swSelect.GetSelectedObjectCount2(-1)
        
            If swSelect.GetSelectedObjectType3(i, -1) = swSelectType_e.swSelFTRFOLDER Then
                
                Dim swFeat As SldWorks.Feature
                Set swFeat = swSelect.GetSelectedObject6(i, -1)
                
                Dim vCompArr() As SldWorks.Component2
                ReDim vCompArr(0)
                
                Call GetComponentsFromFolder(swFeat, vCompArr, 0)
                Call AddComponentsToDictionary(vCompArr, Dict, SelListBox)

            Else
        
                Dim swComp As SldWorks.Component2
                Set swComp = swSelect.GetSelectedObjectsComponent4(i, -1)
                Call CheckAndAddToList(swComp, Dict, SelListBox)
                
            End If

        Next i

    Else

        SelListBox.BackColor = vbRed
        MsgBox "No Components were selected"
    
    End If
    
    swModel.ClearSelection2 True
End Sub

Sub AddComponentsToDictionary(CompArr As Variant, ByRef Dict As Scripting.Dictionary, SelListBox As MSForms.ListBox)

    Dim i As Integer
    For i = LBound(CompArr) To UBound(CompArr)
    
        Dim swComp As SldWorks.Component2
        Set swComp = CompArr(i)
        
        Call CheckAndAddToList(swComp, Dict, SelListBox)
    
    Next i

End Sub

Private Sub CheckAndAddToList(swComp As SldWorks.Component2, ByRef Dict As Scripting.Dictionary, SelListBox As MSForms.ListBox)
    
    If False = swComp.IsSuppressed Then
    
        If Not (Dict.Exists(swComp.Name2)) Then
    
            SelListBox.AddItem
            SelListBox.List(SelListBox.ListCount - 1, 0) = swComp.Name2
            Dict.Add swComp.Name2, swComp
                
        End If
        
    End If
    
End Sub

Private Sub BlockOutFoamSelection_Click()

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swTopLevelModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) = 1 Then

        Set swBlockOutComp = swSelect.GetSelectedObjectsComponent4(1, -1)
        
        If Not swBlockOutComp Is Nothing Then
            
            Dim swBlockOutModel As SldWorks.ModelDoc2
            Set swBlockOutModel = ResolveAndGetModelDoc(swBlockOutComp)
            
            If swBlockOutModel.GetType = swDocumentTypes_e.swDocPART Then

                Me.BlockOutTextBox.Value = "Selected"
                Me.BlockOutTextBox.BackColor = vbGreen

            Else
            
                MsgBox "Warning! Selected component is not a part. Please select the Block Out Foam part", vbCritical, "Selection Warning!"
                
            End If
            
        Else
        
            Me.BlockOutTextBox.Value = "Not Selected"
            Me.BlockOutTextBox.BackColor = vbRed
            
        End If

    ElseIf swSelect.GetSelectedObjectCount2(-1) = 0 Then
        
        MsgBox "Warning! Nothing Selected." & vbCrLf & _
        "Please select Block Out Foam component only", vbCritical, "Selection Warning!"
    
    Else
    
    
        MsgBox "Warning! More than one items are selected." & vbCrLf & _
                "Please select Block Out Foam component only", vbCritical, "Selection Warning!"

    End If
End Sub



Function GetComponentsFromFolder(swFeat As SldWorks.Feature, ByRef TempCompArr As Variant, Level As Integer)

    Dim swFeatFolder As SldWorks.FeatureFolder
    Set swFeatFolder = swFeat.GetSpecificFeature2
            
    Dim i As Integer
    Dim vFeats As Variant
    vFeats = swFeatFolder.GetFeatures
            
    For i = LBound(vFeats) To UBound(vFeats)
                
        Dim compFeat As SldWorks.Feature
        Set compFeat = vFeats(i)
        
        Debug.Print compFeat.Name
                
        If Not (compFeat.GetTypeName2 = "FtrFolder") Then
                
            Dim swComp As SldWorks.Component2
            Set swComp = compFeat.GetSpecificFeature2
                    
            If Not swComp Is Nothing Then
            
                Set TempCompArr(UBound(TempCompArr)) = swComp
                ReDim Preserve TempCompArr(UBound(TempCompArr) + 1)
                
            End If
                
        Else
                
            Call GetComponentsFromFolder(compFeat, TempCompArr, Level + 1)
                             
        End If

    Next i
    
    If Level = 0 Then
    
        ReDim Preserve TempCompArr(UBound(TempCompArr) - 1)
        
    End If
    
End Function
    
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

Private Sub clearFoamListButton_Click()

    Dim i As Integer
    With Me.EpsFoamListBox

        For i = .ListCount - 1 To 0 Step -1
                         
            FoamCompDict.Remove .List(i, 0)
            .RemoveItem (i)
                    
        Next i
        
    End With

End Sub

Private Sub clearRebarListButton_Click()

    Dim i As Integer
    With Me.RebarListBox

        For i = .ListCount - 1 To 0 Step -1
                         
            RebarCompDict.Remove .List(i, 0)
            .RemoveItem (i)
                    
        Next i
        
    End With


End Sub

Private Sub UserForm_QueryClose(Cancel As Integer, CloseMode As Integer)

    Unload DrawingForm

End Sub


Private Sub CreateButton_Click()
 
    Me.Hide
    
    xDirectionVector(0) = 1
    xDirectionVector(1) = 0
    xDirectionVector(2) = 0
    
    yDirectionVector(0) = 0
    yDirectionVector(1) = 1
    yDirectionVector(2) = 0
    
    zDirectionVector(0) = 0
    zDirectionVector(1) = 0
    zDirectionVector(2) = 1

    Dim wallName As String
    wallName = Me.WallNameComboBox.Value
    
    Dim viewName As String
    viewName = GetViewName(wallName)
    
    Dim WeldmentNo As String
    WeldmentNo = Me.WeldNoBox.Value
    
    Dim PFList As IArrListObject
    Set PFList = New IArrListObject
    
    Dim F42List As IArrListObject
    Set F42List = New IArrListObject
    
    Dim LiftingBurkeList As IArrListObject
    Set LiftingBurkeList = New IArrListObject
    
    Dim DowelBarList As IArrListObject
    Set DowelBarList = New IArrListObject
    
    Call GetListOfComponentsWithMatchingVal(PFList, F42List, LiftingBurkeList, DowelBarList)
    
    Debug.Print WeldmentNo
    
'
    Set swMathUtility = swApp.GetMathUtility
    
    Dim swViewNormalVector As SldWorks.MathVector
    Set swViewNormalVector = swMathUtility.CreateVector(GetViewVector(viewName))
'
'    Call ActivateAndRebuildComponent(swConcretePanel)
'    Call ActivateAndRebuildComponent(swTopLevelModel, False)
'    'Call swTopLevelModel.Extension.Rebuild(swRebuildOptions_e.swForceRebuildAll)
'
    Dim swDrawing As SldWorks.DrawingDoc
    Set swDrawing = swApp.NewDocument("C:\FBD\COMMON\FBD Templates\DEFAULT\METAL FAB DRAWING.DRWDOT", 0, 0, 0)
'
    Set swSketchMgr = swDrawing.SketchManager

    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet

    'Call InsertSketchBlock(swDrawing, swSheet, ProjectNo)

    Dim swView As SldWorks.View
    Set swView = swDrawing.CreateDrawViewFromModelView3(swTopLevelModel.GetPathName(), viewName, 0.21593179, 0.17578398, 0)

    Dim oConcreteComp As IComp
    Set oConcreteComp = New IComp

    oConcreteComp.Initialize swConcretePanel, swView

    Dim ViewWidth As Double
    ViewWidth = oConcreteComp.xMax - oConcreteComp.xMin

    Dim ViewHeight As Double
    ViewHeight = oConcreteComp.yMax - oConcreteComp.yMin

    Call ScaleView(swDrawing, swView, ViewWidth, ViewHeight)
    Call UpdateTopViewPosition(oConcreteComp, swDrawing, swView)
    
    Dim swBottomEdge As SldWorks.Edge
    Dim swTopEdge As SldWorks.Edge
    
    Dim swLeftEdge As SldWorks.Edge
    Dim swRightEdge As SldWorks.Edge
    
    Call ConvertAndGetExtremeEdges(swBottomEdge, swTopEdge, swLeftEdge, swRightEdge, swView, oConcreteComp, swViewNormalVector)
    
'
'    Dim FloorPlateList As IArrListObject
'    Set FloorPlateList = GetFloorPlateList(RebarCompDict.Items, swTopView)
'
'    Dim yMinFloorDict As Scripting.Dictionary
'    Set yMinFloorDict = GetConsolidatedDict(FloorPlateList, "yMin", swTopView)
'
'    Dim xMinFloorDict As Scripting.Dictionary
'    Set xMinFloorDict = GetConsolidatedDict(FloorPlateList, "xMin", swTopView)
'
'    Dim xMaxPlateList As IArrListObject
'    Set xMaxPlateList = GetCompListBasedonLocationParam(FloorPlateList, "xMax", swTopView)
'
'    Dim yMaxPlateList As IArrListObject
'    Set yMaxPlateList = GetCompListBasedonLocationParam(FloorPlateList, "yMax", swTopView)
'
'    Dim vBlockOutList As IArrListObject
'    Set vBlockOutList = GetBlockOutList(FloorPlateList.Items, swTopView)
'
'    Dim yMinBlockOutDict As Scripting.Dictionary
'    Set yMinBlockOutDict = GetConsolidatedDict(vBlockOutList, "yMin", swTopView)
'
'    Dim xMinBlockOutDict As Scripting.Dictionary
'    Set xMinBlockOutDict = GetConsolidatedDict(vBlockOutList, "xMin", swTopView)
'
'    Dim ClonedBlockOutList As IArrListObject
'    Set ClonedBlockOutList = vBlockOutList.Clone
'
'    Dim yMaxBlockOutDict As Scripting.Dictionary
'    Set yMaxBlockOutDict = GetConsolidatedDict(ClonedBlockOutList, "yMax", swTopView)
'
'    Dim xMaxBlockOutDict As Scripting.Dictionary
'    Set xMaxBlockOutDict = GetConsolidatedDict(ClonedBlockOutList, "xMax", swTopView)
'
'    Call FindAndAddBeforeBlockOuts(xMaxBlockOutDict, ClonedBlockOutList, "xMin", BlockOutSide_e.Left)
'    Call FindAndAddAfterBlockOuts(xMinBlockOutDict, ClonedBlockOutList, "xMax", BlockOutSide_e.Right)
'
'    Call FindAndAddBeforeBlockOuts(yMaxBlockOutDict, ClonedBlockOutList, "yMin", BlockOutSide_e.Bottom)
'    Call FindAndAddAfterBlockOuts(yMinBlockOutDict, ClonedBlockOutList, "yMax", BlockOutSide_e.Top)
'
'
'    Call SegregateAndAddDimensionVertically(xMinBlockOutDict, xMaxPlateList, xMinFloorDict, oFloorComp, swDrawing, swTopView)
'    Call SegregateAndAddDimensionHorizontally(yMinBlockOutDict, yMaxPlateList, yMinFloorDict, oFloorComp, swDrawing, swTopView)
'
'    Call AddFloorPlateCallouts(xMinFloorDict, swDrawing, swTopView, oFloorComp)
'
'    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, False
'    Call AddCrossMarkAndBalloons(vBlockOutList, swDrawing, swTopView, oFloorComp)
'
'    swDrawing.ClearSelection2 True
'    swTopView.FocusLocked = True
'    Call AddNoteToView(swDrawing, "<FONT size=10PTS style=B>TOP VIEW WITH FLOOR PLATES", _
'        ((oFloorComp.xMax + oFloorComp.xMin) / 2) - 0.025, oFloorComp.yMin - 0.02875)
'
'
'    Dim BottomFloorList As IArrListObject
'    Set BottomFloorList = yMinFloorDict.Items(0)
'    Call AddLocatingHoleDetailView(swDrawing, swTopView, BottomFloorList)
'
'    swTopView.FocusLocked = False
'
'    Call EditTemplate(swDrawing, swDrawing.GetCurrentSheet, WeldmentNo, "FLOOR PLATE LAYOUT")
'    Call AddStructuralNotes(swDrawing)
'    Call SetHiddenEdgesVisibleAndRemoveTangentEdges(swTopView, swDrawing)
'    Call swDrawing.Extension.Rebuild(swRebuildOptions_e.swCurrentSheetDisp)
'
'    Set oFloorComp = Nothing
'    Set swConcretePanel = Nothing
'
'    swApp.SetUserPreferenceToggle swUserPreferenceToggle_e.swSketchInference, True
    
    Unload Me

End Sub

Sub ConvertAndGetExtremeEdges(ByRef swBottomEdge As SldWorks.Edge, ByRef swTopEdge As SldWorks.Edge, _
    ByRef swLeftEdge As SldWorks.Edge, ByRef swRightEdge As SldWorks.Edge, swView As SldWorks.View, _
        oComp As IComp, swViewNormalVector As SldWorks.MathVector)
    
    Set swBottomEdge = GetEdgeInView(oComp, swView, True, False, False)
    Set swTopEdge = GetEdgeInView(oComp, swView, True, True, False)
    Set swLeftEdge = GetEdgeInView(oComp, swView, False, False, False)
    Set swRightEdge = GetEdgeInView(oComp, swView, False, True, False)
    
    Dim swFace As SldWorks.Face2
    
    Dim vFaces As Variant
    vFaces = GetComponentFaces(oComp.GetComponent)

    If Not IsEmpty(vFaces) Then

'        Dim vNormalFaces As Variant
'        vNormalFaces = GetNormalFaces(vFaces, oComp.GetComponent.Transform2, swViewNormalVector)
        
        Set swFace = GetLargestFace(vFaces)

        Call SelectAndConvertEntities(swFace, swView)
        
    End If
    
End Sub

Sub SelectAndConvertEntities(swFace As SldWorks.Face2, swView As SldWorks.View)

    Dim vLoops As Variant
    vLoops = swFace.GetLoops
    
    Dim i As Integer
    For i = LBound(vLoops) To UBound(vLoops)
        
        Dim swLoop As SldWorks.Loop2
        Set swLoop = vLoops(i)
        
        If swLoop.IsOuter Then
        
            Dim vEdges As Variant
            vEdges = swLoop.GetEdges
        
            Dim j As Integer
            For j = LBound(vEdges) To UBound(vEdges)
                
                Dim swEdge As SldWorks.Edge
                Set swEdge = vEdges(j)
                
                swView.SelectEntity swEdge, False
                Call swSketchMgr.SketchUseEdge2(False)
                
            Next j
            
            Exit For
        
        End If
    
    Next i
 
End Sub

Function GetViewVector(viewName As String) As Double()

    Dim vViewRotation As Variant
    vViewRotation = swTopLevelModel.Extension.GetNamedViewRotation(viewName)
    
    Dim swMathVector As SldWorks.MathVector
    Set swMathVector = swMathUtility.CreateVector(zDirectionVector)
    
    Dim vTransformData(15) As Double
    vTransformData(0) = vViewRotation(0)
    vTransformData(1) = vViewRotation(1)
    vTransformData(2) = vViewRotation(2)
    vTransformData(3) = vViewRotation(3)
    vTransformData(4) = vViewRotation(4)
    vTransformData(5) = vViewRotation(5)
    vTransformData(6) = vViewRotation(6)
    vTransformData(7) = vViewRotation(7)
    vTransformData(8) = vViewRotation(8)
    'vTransformData(15) = 1
    
    Dim swMathTransform As SldWorks.MathTransform
    Set swMathTransform = swMathUtility.CreateTransform(vTransformData)
    
    Set swMathVector = swMathVector.MultiplyTransform(swMathTransform.Inverse)
    

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
            
            GetViewVector = GetOppositeVector(yDirectionVector)
            
        Case "*Bottom"
            
            GetViewVector = yDirectionVector
            
        Case Else

           GetViewVector = swMathVector.ArrayData
           
           
    
    End Select
       
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
            
            TempFaces = CombineArr(TempFaces, vFaces)
            
        End If
    
    Next i
    
    GetComponentFaces = TempFaces

End Function
Function CombineArr(ByVal MainArr As Variant, ArrToAdd As Variant)

    Dim i As Integer
    For i = LBound(ArrToAdd) To UBound(ArrToAdd)
    
        ReDim Preserve MainArr(UBound(MainArr) + 1)
        Set MainArr(UBound(MainArr)) = ArrToAdd(i)
        
    Next i
    
    CombineArr = MainArr
    
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
        
        Set swViewNormalVector = swViewNormalVector.Normalise
        
        Dim swFaceNormalVector As SldWorks.MathVector
        Set swFaceNormalVector = swMathUtility.CreateVector(swFace.Normal)
        
        Set swFaceNormalVector = swFaceNormalVector.MultiplyTransform(CompTransform)
        Set swFaceNormalVector = swFaceNormalVector.Normalise
        
        Dim Angle As Double
        Dim DotProduct As Double
        DotProduct = swFaceNormalVector.Dot(swViewNormalVector)
        
        If DotProduct >= 1 Then
        
            Angle = Arccos(Int(DotProduct)) * 180# / 3.14159265359
            
        ElseIf DotProduct < -1 Then
        
            Angle = Arccos(Int(DotProduct) + 1) * 180# / 3.14159265359
        
        Else
        
            Angle = Arccos(DotProduct) * 180# / 3.14159265359
            
        End If
        
        If Not swSurface Is Nothing Then
 
        If swSurface.IsPlane And Angle <= 0.01 Then
            
            Dim swEnt As SldWorks.Entity
            Set swEnt = swFace
            Set swEnt = swEnt.GetSafeEntity
            
            ReDim Preserve NormalFaces(FaceCount)
            Set NormalFaces(FaceCount) = swEnt
            FaceCount = FaceCount + 1
            
        End If
        
        End If

    Next i
    
    GetNormalFaces = NormalFaces

End Function


Sub GetListOfComponentsWithMatchingVal(PFList As IArrListObject, F42List As IArrListObject, _
        LiftingBurkeList As IArrListObject, DowelBarList As IArrListObject)
    
    Dim vComps As Variant
    vComps = swTopLevelModel.GetComponents(True)
    
    Dim i As Integer
    For i = LBound(vComps) To UBound(vComps)
    
        Dim swComp As SldWorks.Component2
        Set swComp = vComps(i)
        
        Dim swCompModel As SldWorks.ModelDoc2
        Set swCompModel = ResolveAndGetModelDoc(swComp)
        
        If Not swCompModel Is Nothing Then
        
            If swCompModel.GetType = swDocumentTypes_e.swDocPART Then
                
                If InStr(swComp.Name2, "198228") > 0 Then
                
                    LiftingBurkeList.AddtoList swComp
                    
                ElseIf InStr(swComp.Name2, "806-11377") > 0 Then
                
                    F42List.AddtoList swComp
                               
                ElseIf InStr(swComp.Name2, "PF") > 0 Then
                
                    PFList.AddtoList swComp
                
                End If
                
            End If
            
        End If
    
    Next i
    
End Sub

Function GetViewName(wallName As String)

    Select Case wallName
        
        Case "Wall-A"
            
            GetViewName = "*Front"
        
        Case "Wall-B"
            
            GetViewName = "*Left"
        
        Case "Wall-C"
        
            GetViewName = "*Back"
        
        Case "Wall-D"
            
            GetViewName = "*Right"
            
        Case "Roof"
            
            GetViewName = "*Top"
            
        Case "Floor"
            
            GetViewName = "*Top"
    
    End Select
    
End Function

Sub ActivateAndRebuildComponent(swComp As Object, Optional ToClose As Boolean = True)
    
    Dim swDoc As SldWorks.ModelDoc2
    Set swDoc = swApp.ActivateDoc3(swComp.GetPathName, True, swRebuildOnActivation_e.swDontRebuildActiveDoc, Err)
    
    If Not swDoc Is Nothing Then
        
        Call swDoc.Extension.Rebuild(swRebuildOptions_e.swForceRebuildAll)
        
        If ToClose Then
        
            swApp.CloseDoc swDoc.GetPathName
            
        End If
        
    End If
    
End Sub

Function GetCompListBasedonLocationParam(ArrList As IArrListObject, Param As String, swView As SldWorks.View) As IArrListObject
    
    Set GetCompListBasedonLocationParam = New IArrListObject
    
    If ArrList.Count > 0 Then
        
        ArrList.SortItems Param
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim keyVal As Double
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oComp As IComp
            Set oComp = vItems(i)
            
            If i = LBound(vItems) Then
                
                keyVal = CallByName(oComp, Param, VbGet)
                GetCompListBasedonLocationParam.AddtoList oComp
                
            Else
            
                If Abs(keyVal - CallByName(oComp, Param, VbGet)) < 0.125 * 0.0254 * swView.ScaleDecimal Then
                
                    GetCompListBasedonLocationParam.AddtoList oComp
                
                End If
            
            End If
        
        Next i

    End If
    
End Function

Function GetConsolidatedDict(MainArrList As IArrListObject, Param As String, swView As SldWorks.View) As Scripting.Dictionary
    
    MainArrList.SortItems Param, False
    Set GetConsolidatedDict = New Scripting.Dictionary
    
    If MainArrList.Count > 0 Then
        
        Dim vComps As Variant
        vComps = MainArrList.Items
        
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim oComp As Object
            Set oComp = vComps(i)

            Dim keyVal As Double
            keyVal = CallByName(oComp, Param, VbGet)
            
            Dim ArrList As IArrListObject
            
            If GetConsolidatedDict.Exists(keyVal) Then

                Set ArrList = GetConsolidatedDict.Item(keyVal)
                ArrList.AddtoList oComp

            Else

                If i = LBound(vComps) Then
                
                    Set ArrList = New IArrListObject
                    ArrList.AddtoList oComp
                    GetConsolidatedDict.Add keyVal, ArrList
                       
                Else
                
                    Dim PrevKeyVal As Double
                    PrevKeyVal = GetConsolidatedDict.Keys(UBound(GetConsolidatedDict.Keys))
                    
                    If Abs(PrevKeyVal - CallByName(oComp, Param, VbGet)) < 0.125 * 0.0254 * swView.ScaleDecimal Then
                        
                       Set ArrList = GetConsolidatedDict.Item(PrevKeyVal)
                       ArrList.AddtoList oComp
                    
                    Else
                        
                        Set ArrList = New IArrListObject
                        ArrList.AddtoList oComp
                        GetConsolidatedDict.Add keyVal, ArrList
                    
                    End If
                
                End If
                
            End If
        
        Next i
        
    End If
    
End Function

Function GetFloorPlateList(vComps As Variant, swView As SldWorks.View) As IArrListObject

    Set GetFloorPlateList = New IArrListObject

    If Not IsEmpty(vComps) Then
        
        Dim i As Integer
        For i = LBound(vComps) To UBound(vComps)
        
            Dim swComp As SldWorks.Component2
            Set swComp = vComps(i)
            
            Dim oComp As IComp
            Set oComp = New IComp
            
            If False = swComp.IsSuppressed Then
            
                oComp.Initialize swComp, swView
                GetFloorPlateList.AddtoList oComp
            
            End If
            
        Next i
        
    End If

End Function

Function GetBlockOutList(vFloorPlates As Variant, swView As SldWorks.View) As IArrListObject
    
    Set GetBlockOutList = New IArrListObject

    If Not IsEmpty(vFloorPlates) Then
    
        Dim i As Integer
        For i = LBound(vFloorPlates) To UBound(vFloorPlates)
        
            Dim oFloorPlate As IComp
            Set oFloorPlate = vFloorPlates(i)
            
            'Debug.Print oFloorPlate.GetComponent.Name2
            
            Dim vFaces As Variant
            vFaces = swView.GetVisibleEntities2(oFloorPlate.GetComponent, swViewEntityType_e.swViewEntityType_Face)
            
            If Not IsEmpty(vFaces) Then
                
                Dim swFace As SldWorks.Face2
            
                If UBound(vFaces) = 0 Then
                
                   Set swFace = vFaces(0)
                   
                Else
                
                    Set swFace = GetLargestFace(vFaces)
                    
                End If
                
                Dim vLoops As Variant
                vLoops = swFace.GetLoops
                
                Set oFloorPlate.VisibleFace = swFace

                Call AddRectangularBlockoutsToList(vLoops, GetBlockOutList, oFloorPlate, swView)

            End If

        Next i
        
    End If
    
End Function

Function GetLargestFace(vFaces As Variant) As SldWorks.Face2

    Dim i As Integer
    Dim Area As Double
    Area = 0
    For i = LBound(vFaces) To UBound(vFaces)
    
        Dim swFace As SldWorks.Face2
        Set swFace = vFaces(i)
        
        If swFace.GetArea > Area Then
        
            Set GetLargestFace = swFace
            Area = swFace.GetArea
            
        End If

    Next i
   
End Function

Sub AddRectangularBlockoutsToList(vLoops As Variant, ArrList As IArrListObject, oComp As IComp, swView As SldWorks.View)

    Dim i As Integer
    For i = LBound(vLoops) To UBound(vLoops)
    
        Dim swLoop As SldWorks.Loop2
        Set swLoop = vLoops(i)
        
        If False = (swLoop.IsOuter) Then
        
            Dim vEdges As Variant
            vEdges = swLoop.GetEdges
            
            If UBound(vEdges) = 3 Then
            
                If Not (IsAnyEdgeHaveNegligibleLength(vEdges)) Then
                
                    If Not (IsContainsCircularEdge(vEdges)) Then
                        
                        Dim oBlockOut As IBlockOut
                        Set oBlockOut = New IBlockOut
    
                        oBlockOut.Initialize swLoop, oComp.GetComponent, swView
                        
                        oComp.AddToBlockOutList oBlockOut
                        ArrList.AddtoList oBlockOut
                    
                    End If
                    
                End If
            
            End If

        End If
    
    Next i

End Sub

Function IsAnyEdgeHaveNegligibleLength(vEdges As Variant) As Boolean

    IsAnyEdgeHaveNegligibleLength = False
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
        
        If GetEdgeLength(swEdge) < 0.1 * 0.0254 Then
            
            IsAnyEdgeHaveNegligibleLength = True
            Exit For
            
        End If

    Next i
    
End Function

Function IsContainsCircularEdge(vEdges As Variant) As Boolean

    IsContainsCircularEdge = False
    
    Dim i As Integer
    For i = LBound(vEdges) To UBound(vEdges)
    
        Dim swEdge As SldWorks.Edge
        Set swEdge = vEdges(i)
        
        Dim swCurve As SldWorks.Curve
        Set swCurve = swEdge.GetCurve
        
        If swCurve.IsCircle Then
        
            IsContainsCircularEdge = True
            Exit For
            
        End If
    
    Next i
    
End Function

Sub AddOrdinateToHorizontalPerimeterBeams(ArrList As IArrListObject, swDrawing As SldWorks.DrawingDoc, _
        swView As SldWorks.View, IsTop As Boolean)
        
    If ArrList.Count > 0 Then
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        swDrawing.ClearSelection2 True
        swDrawing.SetPickMode
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oBeam As IBlockOut
            Set oBeam = vItems(i)
            
            Dim BeamLeftEdge As SldWorks.Edge
            Set BeamLeftEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, False)

            swView.SelectEntity BeamLeftEdge, True

            Dim yPos As Double
        
            If IsTop Then
            
                yPos = oBeam.yMax + 0.01
            
            Else
        
                yPos = oBeam.yMin - 0.01
            
            End If
  
            Dim vComps As Variant

            If IsTop Then
            
                vComps = oBeam.BeforeConnectingPlates
                
            Else
            
                vComps = oBeam.AfterConnectingPlates
                
            End If
            
            
            Call SelectFirstFaceofConnectingPlates(vComps, swView, False)
            
            If i = UBound(vItems) Then
            
                Dim BeamRightEdge As SldWorks.Edge
                Set BeamRightEdge = GetEdgeInViewForBody(oBeam.GetComponent, oBeam, swView, False, True)
            
                swView.SelectEntity BeamRightEdge, True
                
                swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swHorizontalOrdinate, oBeam.xMax, yPos, 0
                Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
                
            End If
        
        
        Next i
    
    End If

End Sub
Private Sub GetViewMaxMinPoints(oComp As IComp, swView As SldWorks.View, ByRef xMin As Double, _
                ByRef xMax As Double, ByRef yMin As Double, ByRef yMax As Double)

    Dim vViewMaxPt As Variant
    vViewMaxPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMaxPointInModel, swView)
            
    Dim vViewMinPt As Variant
    vViewMinPt = GetComponentPointInViewSpace(oComp.GetComponent, oComp.GetMinPointInModel, swView)
    
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(0), vViewMaxPt(0), xMin, xMax)
    Call StrucutralElevation.GetMaxMinPoint(vViewMinPt(1), vViewMaxPt(1), yMin, yMax)
    
End Sub

Function EditTemplate(swDrawing As SldWorks.DrawingDoc, swSheet As SldWorks.Sheet, WeldmentNo As String, SheetName As String)
    
    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swDrawing.SelectionManager()
    
    swDrawing.EditTemplate

    Dim SheetFormatName As String
    SheetFormatName = swSheet.GetSheetFormatName
    Dim Boolstatus As Boolean
    
    Dim swNote As INote
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1227@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
    Set swNote = swSelect.GetSelectedObject6(1, -1)
    swNote.SetText (SheetName)
        
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1229@" & SheetFormatName, "NOTE", 0.355252206998469, 3.32049059009041E-02, 0, False, 0, Nothing, 0)
    Set swNote = swSelect.GetSelectedObject6(1, -1)
    swNote.SetText (WeldmentNo)
    
    Boolstatus = swDrawing.Extension.SelectByID2("DetailItem1262@" & SheetFormatName, "NOTE", 4.69556851111171E-02, 3.48062939323501E-02, 0, False, 0, Nothing, 0)
    swDrawing.EditDelete
        
    swDrawing.EditSheet
    
End Function

Function AddNoteToView(swDrawing As SldWorks.DrawingDoc, NoteText As String, xPos As Double, yPos As Double) As SldWorks.Note
            
    Set AddNoteToView = swDrawing.InsertNote(NoteText)
            
    If Not AddNoteToView Is Nothing Then

        Dim swAnnotation As SldWorks.Annotation
        Set swAnnotation = AddNoteToView.GetAnnotation()

        If Not swAnnotation Is Nothing Then

            swAnnotation.SetPosition xPos, yPos, 0

        End If

    End If
    
End Function
 
Private Sub AddLocatingHoleDetailView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
        BottomFloorPlateList As IArrListObject)

    Dim oFloorPlate As IComp
    Set oFloorPlate = GetPlateWithLeastBlockOuts(BottomFloorPlateList.Items)
    
    Dim CircularEdge As ICircularEdge
    Set CircularEdge = GetBottomCircularEdge(oFloorPlate, swView)
'
'    Dim vCircleParams As Variant
'    vCircleParams = CircularEdge.GetCurve.CircleParams
'
'    Dim dCenterPt(2) As Double
'    dCenterPt(0) = vCircleParams(0)
'    dCenterPt(1) = vCircleParams(1)
'    dCenterPt(2) = vCircleParams(2)
'
'    Dim vCircleCenter As Variant
'    vCircleCenter = dCenterPt
'    vCircleCenter = GetComponentPointInViewSpace(oFloorPlate.GetComponent, vCircleCenter, swView)
    
    Dim dPlateEndPoint(2) As Double
    dPlateEndPoint(0) = oFloorPlate.xMin
    dPlateEndPoint(1) = oFloorPlate.yMin
    dPlateEndPoint(2) = 0
    
    Dim vPlateEndPoint As Variant
    vPlateEndPoint = dPlateEndPoint
    vPlateEndPoint = GetSheetPointInViewSpace(swView, vPlateEndPoint)

    Dim vViewCenterPt(2) As Double
    vViewCenterPt(0) = vPlateEndPoint(0) - 0.125 * 0.0254
    vViewCenterPt(1) = CircularEdge.yViewMin
    vViewCenterPt(2) = 0
            

    Dim radius As Double
    radius = Sqr((vViewCenterPt(0) - CircularEdge.xViewMin) ^ 2 + (vViewCenterPt(1) - CircularEdge.yViewMin) ^ 2) + 4 * 0.0254
            
    Dim swSketchSegment As SldWorks.SketchSegment
    Set swSketchSegment = swSketchMgr.CreateCircleByRadius(vViewCenterPt(0), vViewCenterPt(1), vViewCenterPt(2), radius)

    Dim swDetailView As SldWorks.View
    Set swDetailView = swDrawing.CreateDetailViewAt3(0.32, 0.07, 0, 2, 1, 12, "A", swDetCircleCIRCLE, False)
                    
    If Not swDetailView Is Nothing Then

        Dim swDetailCircle As SldWorks.DetailCircle
        Set swDetailCircle = swDetailView.GetDetail
                    
        swDetailCircle.Layer = "FORMAT"
                
        Dim vDetailOutline As Variant
        vDetailOutline = swDetailView.GetOutline
        
        Dim swDetailLabel As SldWorks.Annotation
        Set swDetailLabel = swDetailView.GetFirstAnnotation3
        
        swDetailLabel.SetPosition2 (vDetailOutline(0) + vDetailOutline(2)) / 2, vDetailOutline(1), 0
        
        Dim Bool As Boolean
        Bool = swDetailView.SelectEntity(CircularEdge.GetEdge, False)

        If Bool Then
        
            Call AddNoteToView(swDrawing, "FLOOR PLATE" & vbCrLf & "LOCATING HOLE", ((vDetailOutline(2) + vDetailOutline(0)) / 2) + radius * swDetailView.ScaleDecimal + 0.0025, ((vDetailOutline(1) + vDetailOutline(3)) / 2) + 0.0025)
        
        End If
        
        
    End If

    
End Sub

Private Function GetBottomCircularEdge(oComp As IComp, swView As SldWorks.View) As ICircularEdge

    Dim vEnts As Variant
    vEnts = GetComponentEdges(oComp.GetComponent)
    
    Dim CircularEdgeList As IArrListObject
    Set CircularEdgeList = New IArrListObject
    
    If Not IsEmpty(vEnts) Then
    
        Dim i As Integer
        For i = LBound(vEnts) To UBound(vEnts)
        
            Dim swEdge As SldWorks.Edge
            Set swEdge = vEnts(i)
            
            Dim swCurve As SldWorks.Curve
            Set swCurve = swEdge.GetCurve
            
            If swCurve.IsCircle Then
            
                Dim vCircleParams As Variant
                vCircleParams = swCurve.CircleParams
                
                If (vCircleParams(6) - 0.125 * 0.0254) <= 0.0001 Then
                    
                    Dim oCircleEdge As ICircularEdge
                    Set oCircleEdge = New ICircularEdge
                    
                    oCircleEdge.Initialize swEdge, vCircleParams, swView, oComp.GetComponent
                    CircularEdgeList.AddtoList oCircleEdge
                    
                End If
                
            End If

        Next i
        
        If CircularEdgeList.Count > 0 Then
        
            CircularEdgeList.SortItems "xMin", False
            CircularEdgeList.SortItems "yMin", False
            
            Set GetBottomCircularEdge = CircularEdgeList.Items(0)
            
        End If
    
    End If
    
End Function

Private Function GetPlateWithLeastBlockOuts(vComps As Variant) As IComp
    
    If Not IsEmpty(vComps) Then
    
        If UBound(vComps) > 0 Then
        
            Dim i As Integer
            
            Dim TempCount As Integer
            TempCount = 1000
            
            For i = 1 To UBound(vComps)
            
                Dim oComp As IComp
                Set oComp = vComps(i)
                
                If oComp.BlockOutList.Count < TempCount Then
                
                    TempCount = oComp.BlockOutList.Count
                    Set GetPlateWithLeastBlockOuts = oComp
                    
                End If
            
            Next i
            
        Else
        
            Set GetPlateWithLeastBlockOuts = vComps(0)
            
        End If
        
    End If

End Function

Private Sub AddHorizontalAssyOrdinate(vBottomBeams As Variant, vTopBeams As Variant, _
    swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)

    swDrawing.ClearSelection2 True
    swDrawing.SetPickMode
    
    Dim BottomBeam As IBlockOut
    Set BottomBeam = vBottomBeams(0)
    
    Dim TopBeam As IBlockOut
    Set TopBeam = vTopBeams(0)
        
    Dim BeamBottomEdge As SldWorks.Edge
    Set BeamBottomEdge = GetEdgeInViewForBody(BottomBeam.GetComponent, BottomBeam, swView, True, False)
        
    Dim BeamTopEdge As SldWorks.Edge
    Set BeamTopEdge = GetEdgeInViewForBody(TopBeam.GetComponent, TopBeam, swView, True, True)
    
    swView.SelectEntity BeamBottomEdge, False
    swView.SelectEntity BeamTopEdge, True

    swDrawing.Extension.AddOrdinateDimension swAddOrdinateDims_e.swVerticalOrdinate, BottomBeam.xMin - 0.01, BottomBeam.yMin, 0
    Call AddBracketsAndSuffixToSelectedDimension(swDrawing, "SEE NOTE 3")
    
End Sub

Private Sub SetHiddenEdgesVisibleAndRemoveTangentEdges(swView As SldWorks.View, swDrawing As SldWorks.DrawingDoc)

    Dim Bool As Boolean
    Bool = swDrawing.Extension.SelectByID2(swView.Name, "DRAWINGVIEW", 0, 0, 0, False, 0, Nothing, swSelectOption_e.swSelectOptionDefault)
    
    If Bool Then
    
        swDrawing.ViewDisplayHiddengreyed
    
    End If
    
    swView.SetDisplayTangentEdges2 swDisplayTangentEdges_e.swTangentEdgesHidden
    
End Sub





Sub UpdateViewLabelPosition(swView As SldWorks.View, yPos As Double)

    Dim swLabelNote As SldWorks.Note
    Set swLabelNote = swView.GetFirstNote
            
    If Not swLabelNote Is Nothing Then
            
        Dim swLabelAnn As SldWorks.Annotation
        Set swLabelAnn = swLabelNote.GetAnnotation
                
        Dim LabelPos As Variant
        LabelPos = swLabelAnn.GetPosition

        swLabelAnn.SetPosition LabelPos(0), yPos, LabelPos(2)
                
    End If
    
End Sub

Function GetViewInASheetByName(swSheet As SldWorks.Sheet, viewName As String) As SldWorks.View

    Dim vViews As Variant
    vViews = swSheet.GetViews
    
    Dim i As Integer
    For i = LBound(vViews) To UBound(vViews)
    
        Dim swView As SldWorks.View
        Set swView = vViews(i)
        
        If swView.Name = viewName Then
            
            Set GetViewInASheetByName = swView
            Exit For
            
        End If
    
    Next i

End Function



Sub AddEllipseAndCreateDetailView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, ArrList As IArrListObject, _
    ByRef LegendAscii As Long, ByRef IsAsciiMaxReached As Boolean, ByRef ViewDict As Scripting.Dictionary, _
    ByRef BodyDict As Scripting.Dictionary, ByRef IsSubWeldmentExists As Boolean)
    
    Dim scaleRatio As Variant
    scaleRatio = swView.scaleRatio

    If ArrList.Count > 0 Then
        
        Dim vItems As Variant
        vItems = ArrList.Items
        
        Dim Clearance As Double
        Clearance = 7.5 * 0.0254 '* swView.ScaleDecimal
        
        Dim i As Integer
        For i = LBound(vItems) To UBound(vItems)
        
            Dim oBlockOut As IBlockOut
            Set oBlockOut = vItems(i)
                   
            If oBlockOut.IsVertical Or False = oBlockOut.IsPerimeter Then
            
                Dim vMinPoint As Variant
                Dim vMaxPoint As Variant
        
                vMinPoint = BodyExtremePointInViewSpace(oBlockOut, swView, False)
                vMaxPoint = BodyExtremePointInViewSpace(oBlockOut, swView, True)
                        
                Dim Width As Double
                Dim vStartPoint(2) As Double
                Dim vEndPoint(2) As Double
        
                If oBlockOut.IsVertical Then
                            
                    Width = Abs(vMaxPoint(0) - vMinPoint(0)) + Clearance
                    vStartPoint(0) = (vMinPoint(0) + vMaxPoint(0)) / 2
                    vStartPoint(1) = vMinPoint(1)
                    vStartPoint(2) = 0
                            
                    vEndPoint(0) = vStartPoint(0)
                    vEndPoint(1) = vMaxPoint(1)
                    vEndPoint(2) = 0
        
                Else
        
                    Width = Abs(vMaxPoint(1) - vMinPoint(1)) + Clearance
                    vStartPoint(0) = vMinPoint(0)
                    vStartPoint(1) = (vMinPoint(1) + vMaxPoint(1)) / 2
                    vStartPoint(2) = 0
                            
                    vEndPoint(0) = vMaxPoint(0)
                    vEndPoint(1) = vStartPoint(1)
                    vEndPoint(2) = 0
                        
                End If
                
                swView.FocusLocked = True
                        
                Dim swSketchSlot As SldWorks.SketchSlot
                Set swSketchSlot = swSketchMgr.CreateSketchSlot(swSketchSlotCreationType_e.swSketchSlotCreationType_line, _
                                swSketchSlotLengthType_e.swSketchSlotLengthType_FullLength, Width, vStartPoint(0), vStartPoint(1), 0, vEndPoint(0), vEndPoint(1), 0, 0, 0, 0, 1, False)
                
                
                
                
                If Not swDetailView Is Nothing Then
                    
                    swDetailView.UseSheetScale = 1
                
                    Dim swDetailCircle As SldWorks.DetailCircle
                    Set swDetailCircle = swDetailView.GetDetail
                    
                    swDetailCircle.Layer = "FORMAT"
                    
                    If False = IsSubWeldmentExists Then
                        
                        IsSubWeldmentExists = True
                    
                    End If
                    
                    ViewDict.Add swDetailView.Name, swDetailView 'oBlockOut.GetBody.Name
                    BodyDict.Add swDetailView.Name, oBlockOut
                    
                End If
                
                Call GetValidAscii(LegendAscii, IsAsciiMaxReached)
                
            End If
            
        Next i
        
    End If

End Sub

Private Sub UpdateTopViewPosition(oFloorComp As IComp, swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View)
    
    Call oFloorComp.CheckForUpdateInMaxMinDimensions(swView)
    
    Dim CenterX As Double
    CenterX = (oFloorComp.xMin + oFloorComp.xMax) / 2
    
    Dim CenterY As Double
    CenterY = (oFloorComp.yMin + oFloorComp.yMax) / 2

    Dim viewPosition As Variant
    viewPosition = swView.Position

    viewPosition(0) = viewPosition(0) + (viewPosition(0) - CenterX)
    viewPosition(1) = viewPosition(1) + (viewPosition(1) - CenterY)

    swView.Position = viewPosition
    
    Call oFloorComp.CheckForUpdateInMaxMinDimensions(swView)

End Sub

Private Function SelectAndAddDimension(swEnt1 As SldWorks.Entity, swEnt2 As SldWorks.Entity, swDrawing As SldWorks.ModelDoc2, _
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

Private Function AddStructuralNotes(swDrawing As SldWorks.DrawingDoc) As SldWorks.Note
    
    Dim swSheet As SldWorks.Sheet
    Set swSheet = swDrawing.GetCurrentSheet
    
    swDrawing.ActivateSheet swSheet.GetName

    Dim swStructuralNote As SldWorks.Note
    Dim Note As String
    
    Note = "<FONT size=10PTS style=B>NOTES:" & vbCrLf & _
            "<FONT size=8PTS style=R>1. DIMENSION ORIGIN STARTING AT LOWER LEFT CORNER OF FLOOR BEAM." & vbCrLf & _
            "2. MAKE SURE THE 1/4" & Chr(34) & " LOCATING HOLES AT WALL-A LOWER LEFT CORNER FOR EACH FLOOR TOP PLATES."

    Set swStructuralNote = swDrawing.CreateText2(Note, 1.99241243641486E-02, 6.92464210842187E-02, 0, 0, 0)
    swStructuralNote.SetTextJustification swTextJustification_e.swTextJustificationLeft

End Function
'
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
                "C:\FBD\COMMON\BLOCKS\" & ProjectNo & " EXTERNAL ELEVATION KEY.SLDBLK", True, 1, 0)


End Sub

Function ScaleView(swDrawing As SldWorks.DrawingDoc, swView As SldWorks.View, _
            ViewWidth As Double, ViewHeight As Double) As SldWorks.View
    

    Dim xScale As Integer
    Dim yScale As Integer
    xScale = GetScaleValue(ViewWidth / (swView.ScaleDecimal * HorizontalMaxDim))
    yScale = GetScaleValue(ViewHeight / (swView.ScaleDecimal * VerticalMaxDim)) '0.20995
    
    Dim IsScaleSet As Boolean
    IsScaleSet = False
    
    If xScale > 0 And yScale > 0 Then
        
        If yScale > xScale Then
            
            IsScaleSet = swView.Sheet.SetScale(1, yScale, True, True)
           
        Else
            
            IsScaleSet = swView.Sheet.SetScale(1, xScale, True, True)
        
        End If
        
    End If

End Function

Function GetScaleValue(scaleVal As Double) As Integer

    GetScaleValue = 0
    
    Dim stdScales As Variant
    stdScales = Array(1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
    
    Dim i As Integer
    For i = LBound(stdScales) To UBound(stdScales)
    
        If stdScales(i) >= scaleVal Then
           GetScaleValue = stdScales(i)
           Exit For
        End If
    
    Next i

End Function


Private Sub UserForm_Initialize()

    Set RebarCompDict = New Scripting.Dictionary
    Set FoamCompDict = New Scripting.Dictionary
    
    With Me.WallNameComboBox
    
        .AddItem "Wall-A"
        .AddItem "Wall-B"
        .AddItem "Wall-C"
        .AddItem "Wall-D"
        .AddItem "Floor"
        .AddItem "Roof"
        
    End With

End Sub

Private Sub PanelSelectionButton_Click()

    Dim swSelect As SldWorks.SelectionMgr
    Set swSelect = swTopLevelModel.SelectionManager
    
    If swSelect.GetSelectedObjectCount2(-1) = 1 Then

        Set swConcretePanel = swSelect.GetSelectedObjectsComponent4(1, -1)
        
        If Not swConcretePanel Is Nothing Then
            
            Dim swConcreteModel As SldWorks.ModelDoc2
            Set swConcreteModel = ResolveAndGetModelDoc(swConcretePanel)
            
            If swConcreteModel.GetType = swDocumentTypes_e.swDocPART Then

                Me.PanelSelectionTextBox.Value = "Selected"
                Me.PanelSelectionTextBox.BackColor = vbGreen

            Else
            
                MsgBox "Warning! Selected component is not a part. Please select the Concrete Panel part", vbCritical, "Selection Warning!"
                
            End If
            
        Else
        
            Me.PanelSelectionTextBox.Value = "Not Selected"
            Me.PanelSelectionTextBox.BackColor = vbRed
            
        End If

    ElseIf swSelect.GetSelectedObjectCount2(-1) = 0 Then
        
        MsgBox "Warning! Nothing Selected." & vbCrLf & _
        "Please select Concrete Panel component only", vbCritical, "Selection Warning!"
    
    Else
    
    
        MsgBox "Warning! More than one items are selected." & vbCrLf & _
                "Please select Concrete Panel component only", vbCritical, "Selection Warning!"

    End If
    
End Sub


