Attribute VB_Name = "ScaleUpdaterModule"
Const BASE_NAME As String = "ScaleUpdater"

Dim swPostGenList As PostRegenerateListener

Sub Main()

    Dim swApp As SldWorks.SldWorks
    Set swApp = Application.SldWorks
    
    Dim swModel As SldWorks.ModelDoc2
    
    Set swModel = swApp.ActiveDoc
    
    If Not swModel Is Nothing Then
        
        If swModel.GetType() = swDocumentTypes_e.swDocDRAWING Then

            Dim curMacroPath As String
            curMacroPath = swApp.GetCurrentMacroPathName
                
            Dim vMethods(8) As String
            Dim moduleName As String
                
            GetMacroEntryPoint swApp, curMacroPath, moduleName, ""
                
            vMethods(0) = curMacroPath: vMethods(1) = moduleName: vMethods(2) = "swmRebuild"
            vMethods(3) = curMacroPath: vMethods(4) = moduleName: vMethods(5) = "swmEditDefinition"
            vMethods(6) = curMacroPath: vMethods(7) = moduleName: vMethods(8) = "swmSecurity"
                
            Dim swFeat As SldWorks.Feature
            Set swFeat = swModel.FeatureManager.InsertMacroFeature3(BASE_NAME, "", vMethods, _
                    Empty, Empty, Empty, Empty, Empty, Empty, _
                    Empty, swMacroFeatureOptions_e.swMacroFeatureEmbedMacroFile)
                
            If swFeat Is Nothing Then
            
                MsgBox "Failed to create ScaleUpdater Feature"
                
            End If
                
        End If
        
    Else
    
        MsgBox "Please open model"
        
    End If
    
End Sub

Sub GetMacroEntryPoint(app As SldWorks.SldWorks, macroPath As String, ByRef moduleName As String, ByRef procName As String)
        
    Dim vMethods As Variant
    vMethods = app.GetMacroMethods(macroPath, swMacroMethods_e.swMethodsWithoutArguments)
    
    Dim i As Integer
    
    If Not IsEmpty(vMethods) Then
    
        For i = 0 To UBound(vMethods)
            Dim vData As Variant
            vData = Split(vMethods(i), ".")
            
            If i = 0 Or LCase(vData(1)) = "main" Then
                moduleName = vData(0)
                procName = vData(1)
            End If
        Next
        
    End If
    
End Sub

Function swmRebuild(varApp As Variant, varDoc As Variant, varFeat As Variant) As Variant

    Dim swApp As SldWorks.SldWorks
    Dim swModel As SldWorks.ModelDoc2
    Dim swFeat As SldWorks.Feature
    
    Set swApp = varApp
    Set swModel = varDoc
    Set swFeat = varFeat
    
    Dim swMacroFeat As SldWorks.MacroFeatureData
    Set swMacroFeat = swFeat.GetDefinition()

    If swPostGenList Is Nothing Then
        Set swPostGenList = New PostRegenerateListener
    End If
        
    swPostGenList.Init swApp, swModel

End Function

Function swmEditDefinition(varApp As Variant, varDoc As Variant, varFeat As Variant) As Variant
    swmEditDefinition = True
End Function

Function swmSecurity(varApp As Variant, varDoc As Variant, varFeat As Variant) As Variant
    swmSecurity = SwConst.swMacroFeatureSecurityOptions_e.swMacroFeatureSecurityByDefault
End Function


