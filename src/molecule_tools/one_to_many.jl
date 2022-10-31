using Distributed, ClusterManagers

function setup_environment(num_workers::Int, include_files::Union{Vector{String}, Nothing}=nothing, cluster_type::Symbol=:none)
    if cluster_type === :none
        addprocs(num_workers)
    elseif cluster_type == :slurm
        addprocs_slurm(num_workers)
    end

    if include_files !== nothing
        for include_files in include_files
            @everywhere include(include_file)
        end
    end
end

function one_to_many(jobs::Vector{Function})
    """
    This function runs a set of jobs which should be distributed across
    the available workers. These may be local processes or processes on
    another node. This function assumes that nothing needs to be done other
    than manage the submission of the jobs to other nodes. That is, no
    attempt is made to receive data from the nodes, so whatever calculation
    is done, the output needs to be logged by the worker.
    """
    worker_array = workers()
    active_jobs::Vector{Union{Future, Nothing}}(nothing, length(worker_array))
    current_job_index = 1
    for i in eachindex(worker_array)
        if i < length(jobs)
            @async begin
                active_jobs[i] = @spawnat worker_array[i] jobs[current_job_index]
                current_job_index += 1
            end
        end
    end

    completed_jobs = 0
    num_jobs = length(jobs)
    while true
        for i in eachindex(worker_array)
            # check if currently running jobs is complete
            # if it is submit the next job immediately
            if jobs[i] !== nothing
                if isready(active_jobs[i])
                    completed_jobs += 1
                    if current_job_index < num_jobs
                        active_jobs[i] = @spawnat worker_array[i] jobs[current_job_index]()
                        current_job_index += 1
                    end
                end
            end
        end
        if completed_jobs == num_jobs
            break
        end
    end
end