#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["MLConsoleApp/MLConsoleApp.csproj", "MLConsoleApp/"]
RUN dotnet restore "MLConsoleApp/MLConsoleApp.csproj"
COPY . .
WORKDIR "/src/MLConsoleApp"
RUN dotnet build "MLConsoleApp.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MLConsoleApp.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MLConsoleApp.dll"]
