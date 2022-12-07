using PIQL
using Documenter

DocMeta.setdocmeta!(PIQL, :DocTestSetup, :(using PIQL); recursive=true)

makedocs(;
    modules=[PIQL],
    authors="chelate <42802644+chelate@users.noreply.github.com> and contributors",
    repo="https://github.com/chelate/PIQL.jl/blob/{commit}{path}#{line}",
    sitename="PIQL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chelate.github.io/PIQL.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chelate/PIQL.jl",
    devbranch="main",
)
