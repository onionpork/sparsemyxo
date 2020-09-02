function est_matrix = funkSVD(rating_mat, latent_features, learning_rate, iters)
n_t = size(rating_mat, 1);
n_species = size(rating_mat, 2);
%num_ratings = n_t* n_species - sum(sum(isnan(rating_mat)));

t_matrix = rand(n_t, latent_features);
species_matrix = rand(latent_features, n_species);

sse_initial = 0;

for iteration = 1:iters
    old_sse = sse_initial;
    disp(old_sse)
    sse_initial = 0;
    
    for i = 1:n_t
        for j  = 1:n_species
            if ~isnan(rating_mat(i, j))
                diff = rating_mat(i,j) - t_matrix(i,:) * species_matrix(:,j);
                sse_initial = sse_initial + diff^2;
                
                for k = 1:latent_features
                    t_matrix(i, k) = t_matrix(i,k) ...
                        + learning_rate*(2*diff*species_matrix(k,j));
                    species_matrix(k,j) = species_matrix(k,j) ...
                        + learning_rate*(2*diff*t_matrix(i,k));
                end
            end
        end
    end
    
est_matrix = t_matrix*species_matrix;
     

end