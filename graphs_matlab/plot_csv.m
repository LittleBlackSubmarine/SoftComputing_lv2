% Elites changing
plot(original(1,:), original(2,:));
hold on;
plot(elites(1,:), elites(2,:));
hold on;
plot(elites(3,:),elites(4,:));
hold on;
plot(elites(5,:),elites(6,:));
hold off;

title('Fitness = f(Elites)')
xlabel('Generations') 
ylabel('Fitness') 
legend('Elites: 5','Elites: 10','Elites: 15','Elites: 20')

% Mutation changing
mutation_fig = figure;

plot(original(1,:), original(2,:));
hold on;
plot(mutation(1,:), mutation(2,:));
hold on;
plot(mutation(3,:), mutation(4,:));
hold on;
plot(mutation(5,:), mutation(6,:));
hold off;

title('Fitness = f(Mutation)')
xlabel('Generations') 
ylabel('Fitness') 
legend('Mutation: 5%','Mutation: 10%','Mutation: 15%','Mutation: 20%')

% Population changing
population_fig = figure;

plot(original(1,:), original(2,:));
hold on;
plot(population(1,:),population(2,:));
hold on;
plot(population(3,:),population(4,:));
hold on;
plot(population(5,:),population(6,:));
hold off;

title('Fitness = f(Population)')
xlabel('Generations') 
ylabel('Fitness') 
legend('Population: 50','Population: 100','Population: 200','Population: 400')

